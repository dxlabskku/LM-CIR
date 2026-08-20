import os
import json
from typing import Any, Dict, List, Optional

import torch
import open_clip
from tqdm import tqdm


FEATURE_PTH = "LM-CIR/image_features/feature_L-14/cirr_test1_feature.pt"
QUERY_JSON = "LM-CIR/generated_captions/COT_cirr_test1.json"
# Each pairid maps to the candidate members ranked for that query.
SUBSET_CAP_JSON = "CIRR/data/cirr/captions/cap.rc2.test1.json"
OUTPUT_JSON = "LM-CIR/first_stage_result/submit/COT_cirr_gpt_51_fu_02_subset.json"

pretraining = {
    "ViT-B-32": "laion2b_s34b_b79k",
    "ViT-L-14": "laion2b_s32b_b82k",
    "ViT-bigG-14": "laion2b_s39b_b160k"
}
MODEL_NAME = "ViT-L-14"

SET_alpha = 1.0
SET_beta = 0.2


def get_stem(path_or_name: str) -> str:
    """Return a filename without its directory or extension."""
    base = os.path.basename(path_or_name)
    stem, _ = os.path.splitext(base)
    return stem


class CIRRTextRetriever:
    def __init__(
        self,
        feature_pth: str,
        model_name: str = MODEL_NAME,
        pretrained: str = pretraining[MODEL_NAME],
        device: Optional[str] = None,
        text_batch_size: int = 256,
        sim_chunk_size: int = 50000,
        topk: int = 51,
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
        db = torch.load(feature_pth, map_location="cpu", weights_only=False)

        if "features" not in db or "image_names" not in db:
            raise KeyError("Feature file must contain the keys: features and image_names")

        self.image_feats = db["features"]
        self.image_names = db["image_names"]

        if not isinstance(self.image_names, list):
            self.image_names = list(self.image_names)

        if self.image_feats.shape[0] != len(self.image_names):
            raise ValueError(
                f"Feature rows ({self.image_feats.shape[0]}) do not match the number of image names ({len(self.image_names)})"
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
                f"Duplicate stems prevent unique reference mapping: {len(set(duplicate_stems))} duplicates"
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
        """Build one reference-image feature per query sample."""
        ref_feats = []

        for item in samples:
            ref_stem = None
            ref_path = item.get("reference_path", None)
            if ref_path is not None and str(ref_path).strip() != "":
                rp = ref_path[0] if isinstance(ref_path, list) else ref_path
                if rp is not None and str(rp).strip():
                    ref_stem = get_stem(str(rp))
            elif item.get("reference_id") not in (None, ""):
                ref_stem = str(item["reference_id"])

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

        return torch.stack(ref_feats, dim=0)

    @torch.no_grad()
    def search_topk_for_text_and_ref_feats(
        self,
        text_feats: torch.Tensor,
        ref_feats: torch.Tensor,
        alpha: float = SET_alpha,
        beta: float = SET_beta,
    ):
        """Compute chunked top-k results using text-reference fusion."""
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
            sims = alpha * text_sims + beta * ref_sims

            chunk_k = min(K, end - start)
            chunk_scores, chunk_local_idx = torch.topk(sims, k=chunk_k, dim=1)
            chunk_global_idx = chunk_local_idx + start

            # Merge this chunk's candidates into the running global top-k.
            merged_scores = torch.cat([global_top_scores, chunk_scores], dim=1)
            merged_indices = torch.cat([global_top_indices, chunk_global_idx], dim=1)

            new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1)
            new_indices = torch.gather(merged_indices, 1, new_pos)

            global_top_scores = new_scores
            global_top_indices = new_indices

        return global_top_scores, global_top_indices

    @torch.no_grad()
    def rank_members_in_subset_fusion(
        self,
        text_feat_1d: torch.Tensor,
        ref_feat_1d: torch.Tensor,
        member_ids: List[str],
        alpha: float = SET_alpha,
        beta: float = SET_beta,
    ) -> List[str]:
        """Rank candidate members by fused text and reference similarity."""
        if len(member_ids) == 0:
            return []

        gallery_indices: List[int] = []
        for mid in member_ids:
            stem = get_stem(str(mid))
            gi = self.stem_to_index.get(stem)
            if gi is None:
                raise KeyError(
                    f"Subset member {mid!r} (stem={stem!r}) is not present in image_names"
                )
            gallery_indices.append(gi)

        idx_t = torch.tensor(gallery_indices, dtype=torch.long, device=self.device)
        subset_feats = self.image_feats[idx_t]

        tf = text_feat_1d.to(self.device, dtype=torch.float32)
        rf = ref_feat_1d.to(self.device, dtype=torch.float32)
        if tf.dim() == 1:
            tf = tf.unsqueeze(0)
        if rf.dim() == 1:
            rf = rf.unsqueeze(0)

        text_sims = (tf @ subset_feats.T).squeeze(0)
        ref_sims = (rf @ subset_feats.T).squeeze(0)
        sims = alpha * text_sims + beta * ref_sims

        order = torch.argsort(sims, descending=True).cpu().tolist()
        return [str(member_ids[j]) for j in order]


def main():
    print("[Info] Loading query json...")
    with open(QUERY_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("Query JSON must have the format list[dict]")

    print(f"[Info] Num queries = {len(samples)}")

    print("[Info] Loading subset caption json (pairid -> img_set.members)...")
    with open(SUBSET_CAP_JSON, "r", encoding="utf-8") as f:
        cap_items = json.load(f)
    if not isinstance(cap_items, list):
        raise ValueError("Subset caption JSON must have the format list[dict]")
    pairid_to_members: dict = {}
    for row in cap_items:
        if "pairid" not in row or "img_set" not in row:
            raise KeyError("Subset entry is missing pairid or img_set")
        pid = str(row["pairid"])
        members = row["img_set"].get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"pairid={pid} has invalid img_set.members")
        pairid_to_members[pid] = [str(m) for m in members]
    print(f"[Info] Loaded subset definitions for {len(pairid_to_members)} pairids")

    edited_captions = []
    for i, item in enumerate(samples):
        if "edited_caption" not in item:
            raise KeyError(f"Sample {i} is missing edited_caption")
        edited_captions.append(item["edited_caption"])

    retriever = CIRRTextRetriever(
        feature_pth=FEATURE_PTH,
        model_name=MODEL_NAME,
        pretrained=pretraining[MODEL_NAME],
        device=None,
        text_batch_size=256,
        sim_chunk_size=50000,
        topk=51,
    )

    text_feats = retriever.encode_texts(edited_captions)
    ref_feats = retriever.build_reference_feats(samples)

    print("[Info] Building submission json (subset-only text+ref fusion ranking)...")
    submission: dict = {
        "version": "rc2",
        "metric": "recall_subset",
    }

    for i, item in enumerate(tqdm(samples, desc="Formatting results")):
        if "pairid" not in item:
            raise KeyError(f"Sample {i} is missing pairid")
        pairid = str(item["pairid"])

        members = pairid_to_members.get(pairid)
        if members is None:
            raise KeyError(
                f"pairid={pairid} has no matching members in {SUBSET_CAP_JSON}; check that the datasets are aligned"
            )

        tf1 = text_feats[i]
        rf1 = ref_feats[i]
        ranked_members = retriever.rank_members_in_subset_fusion(
            tf1,
            rf1,
            members,
            alpha=SET_alpha,
            beta=SET_beta,
        )

        ref_stem = None
        ref_path = item.get("reference_path", None)
        if ref_path is not None and str(ref_path).strip() != "":
            rp = ref_path[0] if isinstance(ref_path, list) else ref_path
            if rp is not None and str(rp).strip():
                ref_stem = get_stem(str(rp))
        elif item.get("reference_id") not in (None, ""):
            ref_stem = str(item["reference_id"])

        if ref_stem:
            ref_stem_norm = get_stem(ref_stem)
            ranked_members = [m for m in ranked_members if get_stem(m) != ref_stem_norm]

        if len(ranked_members) != len(members) - 1:
            raise ValueError(
                f"pairid={pairid}: expected {len(members) - 1} results after removing the reference, but got {len(ranked_members)}"
            )

        submission[pairid] = ranked_members[:]

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    print(f"[Info] Saved submission to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
