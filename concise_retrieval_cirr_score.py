# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Any, Optional

import torch
import open_clip
from tqdm import tqdm


# Configure the input and output paths here.
FEATURE_PTH = "LM-CIR/image_features/feature_L-14/cirr_test1_feature.pt"
QUERY_JSON = "LM-CIR/generated_captions/COT_cirr_test1.json"
OUTPUT_JSON = "LM-CIR/first_stage_result/score/COT_cirr_test1.json"

pretraining = {
    "ViT-B-32": "laion2b_s34b_b79k",
    "ViT-L-14": "laion2b_s32b_b82k",
    "ViT-bigG-14": "laion2b_s39b_b160k"
}
MODEL_NAME = "ViT-L-14"

SET_alpha = 1.0
SET_beta = 0.2

def get_stem(path_or_name: str) -> str:
    base = os.path.basename(path_or_name)
    stem, _ = os.path.splitext(base)
    return stem

def safe_float(x) -> float:
    return float(x)

class CIRRTextRetriever:
    def __init__(
        self,
        feature_pth: str,
        model_name: str = MODEL_NAME,
        pretrained: str = pretraining[MODEL_NAME],
        device: Optional[str] = None,
        text_batch_size: int = 256,
        sim_chunk_size: int = 50000,
        topk: int = 101,
    ):
        self.feature_pth = feature_pth
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        self.text_batch_size = text_batch_size
        self.sim_chunk_size = sim_chunk_size
        self.topk = topk

        print(f"[Info] device = {self.device}")
        print("[Info] Loading feature database...")
        db = torch.load(feature_pth, map_location="cpu")

        if "features" not in db or "image_names" not in db:
            raise KeyError("feature pth must contain keys: 'features' and 'image_names'")

        self.image_feats = db["features"]
        self.image_names = db["image_names"]

        if not isinstance(self.image_names, list):
            self.image_names = list(self.image_names)

        if self.image_feats.shape[0] != len(self.image_names):
            raise ValueError(
                f"features rows {self.image_feats.shape[0]} != image_names count {len(self.image_names)}"
            )

        self.image_feats = self.image_feats.to(torch.float32).contiguous()

        self.num_images, self.feat_dim = self.image_feats.shape
        print(f"[Info] num_images = {self.num_images}")
        print(f"[Info] feat_dim   = {self.feat_dim}")

        self.stem_to_index = {}
        duplicate_stems = []
        for idx, name in enumerate(self.image_names):
            stem = get_stem(name)
            if stem in self.stem_to_index:
                duplicate_stems.append(stem)
            self.stem_to_index[stem] = idx

        if duplicate_stems:
            raise ValueError(
                f"Duplicate stems detected, cannot uniquely map target. Duplicate count={len(set(duplicate_stems))}"
            )

        print("[Info] Loading open_clip model...")
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        if self.device == "cuda":
            self.image_feats = self.image_feats.to(self.device, non_blocking=True)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        all_text_feats = []

        for start in tqdm(range(0, len(texts), self.text_batch_size), desc="Encoding texts", ascii=True):
            batch_texts = texts[start:start + self.text_batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)

            txt_feat = self.model.encode_text(tokens)
            txt_feat = txt_feat.to(torch.float32)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            all_text_feats.append(txt_feat)

        text_feats = torch.cat(all_text_feats, dim=0)
        return text_feats
    
    @torch.no_grad()
    def build_reference_feats(
        self,
        samples: List[Dict[str, Any]],
    ) -> torch.Tensor:
        ref_feats = []

        for item in samples:
            ref_path = item.get("reference_path", None)
            ref_stem = None

            if ref_path is not None:
                rp = ref_path[0] if isinstance(ref_path, list) else ref_path
                if rp is not None and str(rp).strip():
                    ref_stem = get_stem(str(rp))

            if ref_stem is None:
                ref_feat = torch.zeros(self.feat_dim, dtype=torch.float32, device=self.device)
            else:
                ref_idx = self.stem_to_index.get(ref_stem, None)
                if ref_idx is None:
                    print(f"[Warning] reference not found in feature db: {ref_stem}")
                    ref_feat = torch.zeros(self.feat_dim, dtype=torch.float32, device=self.device)
                else:
                    ref_feat = self.image_feats[ref_idx]

            ref_feats.append(ref_feat)

        ref_feats = torch.stack(ref_feats, dim=0)
        return ref_feats

    @torch.no_grad()
    def search_topk_for_text_and_ref_feats(
        self,
        text_feats: torch.Tensor,
        ref_feats: torch.Tensor,
        alpha: float = SET_alpha,
        beta: float = SET_beta,
    ):
        B = text_feats.shape[0]
        K = self.topk
        N = self.num_images

        global_top_scores = torch.full(
            (B, K),
            fill_value=-1e9,
            dtype=torch.float32,
            device=self.device,
        )
        global_top_indices = torch.full(
            (B, K),
            fill_value=-1,
            dtype=torch.long,
            device=self.device,
        )

        for start in tqdm(range(0, N, self.sim_chunk_size), desc="Similarity chunks", leave=False):
            end = min(start + self.sim_chunk_size, N)
            img_chunk = self.image_feats[start:end]

            text_sims = text_feats @ img_chunk.T
            ref_sims = ref_feats @ img_chunk.T
            # Combine text and reference-image similarities.
            sims = alpha * text_sims + beta * ref_sims

            chunk_k = min(K, end - start)
            chunk_scores, chunk_local_idx = torch.topk(sims, k=chunk_k, dim=1)
            chunk_global_idx = chunk_local_idx + start

            merged_scores = torch.cat([global_top_scores, chunk_scores], dim=1)
            merged_indices = torch.cat([global_top_indices, chunk_global_idx], dim=1)

            new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1)
            new_indices = torch.gather(merged_indices, 1, new_pos)

            global_top_scores = new_scores
            global_top_indices = new_indices

        return global_top_scores, global_top_indices

    @torch.no_grad()
    def compute_target_in_topk(
        self,
        top_indices: torch.Tensor,
        top_scores: torch.Tensor,
        target_stem: str,
    ):
        target_idx = self.stem_to_index.get(target_stem, None)

        if target_idx is None:
            return {
                "target_name": None,
                "target_id": target_stem,
                "target_score": None,
                "target_rank": None,
                "target_found_in_feature_db": False,
            }

        top_indices_list = top_indices.tolist()

        if target_idx in top_indices_list:
            rank = top_indices_list.index(target_idx) + 1
            score = top_scores[rank - 1].item()

            return {
                "target_name": self.image_names[target_idx],
                "target_id": target_stem,
                "target_score": float(score),
                "target_rank": rank,
                "target_found_in_feature_db": True,
            }
        else:
            return {
                "target_name": self.image_names[target_idx],
                "target_id": target_stem,
                "target_score": None,
                "target_rank": ">100",
                "target_found_in_feature_db": True,
            }


def main():
    print("[Info] Loading query json...")
    with open(QUERY_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("my_cirr_val.json should be in list[dict] format")

    print(f"[Info] Num queries = {len(samples)}")

    edited_captions = []
    for i, item in enumerate(samples):
        if "edited_caption" not in item:
            raise KeyError(f"Sample {i} is missing 'edited_caption'")
        edited_captions.append(item["edited_caption"])

    retriever = CIRRTextRetriever(
        feature_pth=FEATURE_PTH,
        model_name=MODEL_NAME,
        pretrained=pretraining[MODEL_NAME],
        device=None,
        text_batch_size=256,
        sim_chunk_size=50000,
        topk=101,
    )

    text_feats = retriever.encode_texts(edited_captions)
    ref_feats = retriever.build_reference_feats(samples)

    print("[Info] Searching top101...")
    top_scores, top_indices = retriever.search_topk_for_text_and_ref_feats(
        text_feats=text_feats,
        ref_feats=ref_feats,
        alpha=SET_alpha,
        beta=SET_beta,
    )

    top_scores = top_scores.cpu()
    top_indices = top_indices.cpu()
    text_feats_cpu = text_feats.detach().cpu()

    print("[Info] Building output json...")
    output_samples = []

    for i, item in enumerate(tqdm(samples, desc="Formatting results")):
        new_item = dict(item)

        cur_scores = top_scores[i].tolist()
        cur_indices = top_indices[i].tolist()
        pairs: List[tuple] = list(zip(cur_scores, cur_indices))

        ref_path = item.get("reference_path")
        ref_stem = None
        if ref_path is not None:
            rp = ref_path[0] if isinstance(ref_path, list) else ref_path
            if rp is not None and str(rp).strip():
                ref_stem = get_stem(str(rp))

        if ref_stem:
            ref_idx = retriever.stem_to_index.get(ref_stem)
            if ref_idx is not None:
                pairs = [(s, idx) for s, idx in pairs if idx != ref_idx]

        pairs = pairs[:100]

        top_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        top_score = torch.tensor([p[0] for p in pairs], dtype=torch.float32)

        cur_top50 = []
        text_feat_i = text_feats[i].detach().cpu()
        ref_feat_i = ref_feats[i].detach().cpu()

        for rank, (score, idx) in enumerate(pairs[:50], start=1):
            image_name = retriever.image_names[idx]
            image_id = get_stem(image_name)

            cand_feat = retriever.image_feats[idx].detach().cpu()
            text_score = torch.dot(text_feat_i, cand_feat).item()
            image_score = torch.dot(ref_feat_i, cand_feat).item()
            final_score = SET_alpha * text_score + SET_beta * image_score

            cur_top50.append({
                "rank": rank,
                "image_name": image_name,
                "image_id": image_id,
                "score": safe_float(final_score),
                "text_score": safe_float(text_score),
                "image_score": safe_float(image_score),
            })

        new_item["retrieval_top50"] = cur_top50

        target_info = None
        target_path = item.get("target_path", None)

        if target_path is not None:

            if isinstance(target_path, list):
                target_results = []

                for p in target_path:
                    if str(p).strip() == "":
                        continue

                    target_stem = get_stem(p)

                    target_info = retriever.compute_target_in_topk(
                        top_indices=top_idx,
                        top_scores=top_score,
                        target_stem=target_stem,
                    )

                    target_results.append(target_info)

                new_item["target_results"] = target_results

            elif str(target_path).strip() != "":
                target_stem = get_stem(target_path)

                target_info = retriever.compute_target_in_topk(
                    top_indices=top_idx,
                    top_scores=top_score,
                    target_stem=target_stem,
                )

                new_item["target_result"] = target_info

        elif item.get("target_id", None):
            target_stem = str(item["target_id"])

            target_info = retriever.compute_target_in_topk(
                top_indices=top_idx,
                top_scores=top_score,
                target_stem=target_stem,
            )

        if target_info is not None and "target_results" not in new_item:
            new_item["target_result"] = target_info

        output_samples.append(new_item)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)

    print(f"[Info] Saved results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()