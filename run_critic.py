# -*- coding: utf-8 -*-
"""
Read baseline result JSON and rerank top-N candidates using a VLM critic.

Usage:
  python run_critic.py \
    --baseline ./baselines/fiq-shirt_val_baseline_fu030.json \
    --output   ./results/fiq-shirt_val_critic.json \
    --topn 20 --w_baseline 0.0 --w_edit 1.0 --w_preserve 1.0 \
"""
import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from critic import VLMCritic

def get_stem(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name))[0]

def fix_path(p):
    """Override this function for your environment if needed."""
    return p


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--topn",     type=int, default=10)
    p.add_argument("--w_baseline", type=float, default=0.0)
    p.add_argument("--w_edit",   type=float, default=1.0)
    p.add_argument("--w_preserve", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=0,
                   help="0=all, otherwise limit query count")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="Shard index (0-based)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards (>=1). Processes samples[shard_idx::num_shards]")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-32B-Instruct")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_cache(path: str) -> dict:
    cache = {}
    p = Path(path)
    if p.exists():
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cache[(row["pairid"], row["cand_id"])] = row
    return cache


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    base_stem = Path(args.baseline).stem
    if args.num_shards > 1:
        cache_file = f"{args.cache_dir}/{base_stem}.shard{args.shard_idx}of{args.num_shards}.jsonl"
    else:
        cache_file = f"{args.cache_dir}/{base_stem}.jsonl"
    # Also load shared cache (reuse from previous single-shard runs)
    shared_cache_file = f"{args.cache_dir}/{base_stem}.jsonl"

    cache = load_cache(cache_file)
    if shared_cache_file != cache_file and Path(shared_cache_file).exists():
        shared = load_cache(shared_cache_file)
        for k, v in shared.items():
            cache.setdefault(k, v)
    cache_fp = open(cache_file, "a")

    print(f"[Info] baseline   : {args.baseline}")
    print(f"[Info] output     : {args.output}")
    print(f"[Info] cache      : {cache_file}  (loaded {len(cache)} entries)")
    print(f"[Info] shard      : {args.shard_idx}/{args.num_shards}")
    print(f"[Info] topN       : {args.topn}")
    print(f"[Info] weights    : baseline={args.w_baseline} edit={args.w_edit} preserve={args.w_preserve}")
    print(f"[Info] model      : {args.model_id}  device={args.device}")

    with open(args.baseline, encoding="utf-8") as f:
        all_samples = json.load(f)
    if args.max_samples:
        all_samples = all_samples[: args.max_samples]
    # Shard slicing: samples[shard_idx::num_shards]
    samples = all_samples[args.shard_idx :: args.num_shards]
    print(f"[Info] N total = {len(all_samples)}, this shard = {len(samples)}")

    print("[Info] Loading critic model (this may take a while)...")
    critic = VLMCritic(model_id=args.model_id, device_map=args.device)
    print("[Info] Critic ready.")

    out = []
    n_calls = 0
    n_cached = 0

    for item in tqdm(samples, desc="Reranking"):
        pairid = str(item.get("pairid", item.get("reference_id", "")))
        ref_path_raw = item.get("reference_path")
        if isinstance(ref_path_raw, list):
            ref_path_raw = ref_path_raw[0] if ref_path_raw else None
        ref_path = fix_path(ref_path_raw)
        # Use directory of reference_path as fallback when only image_name is available
        image_dir = os.path.dirname(ref_path) if ref_path else None

        # Modification text: prefer instruction if available, fallback to edited_caption
        mod_text = item.get("instruction") or item.get("edited_caption", "")

        cands = (item.get("retrieval_top50") or item.get("retrieval_top10") or [])[: args.topn]

        scored = []
        for c in cands:
            cand_id = c.get("image_id") or get_stem(c.get("image_name", ""))
            cand_path = c.get("image_path")
            if cand_path:
                cand_path = fix_path(cand_path)
            elif image_dir and c.get("image_name"):
                cand_path = os.path.join(image_dir, c["image_name"])
            else:
                cand_path = fix_path(c.get("image_name"))

            key = (pairid, cand_id)
            if key in cache:
                cs = cache[key]
                n_cached += 1
            else:
                try:
                    r = critic.score(ref_path, mod_text, cand_path)
                except Exception as e:
                    r = {"edit": 0, "preserve": 0, "reason": f"ERROR: {e}", "raw": ""}
                cs = {"pairid": pairid, "cand_id": cand_id, **r}
                cache[key] = cs
                cache_fp.write(json.dumps(cs) + "\n")
                cache_fp.flush()
                n_calls += 1

            new_score = (
                args.w_baseline * float(c.get("score", 0.0))
                + args.w_edit   * float(cs["edit"])
                + args.w_preserve * float(cs["preserve"])
            )
            scored.append((new_score, c, cs))

        scored.sort(key=lambda x: -x[0])

        new_item = dict(item)
        new_item["retrieval_top10_critic"] = [
            {
                **c,
                "critic_edit": cs["edit"],
                "critic_preserve": cs["preserve"],
                "critic_reason": cs.get("reason", ""),
                "rerank_score": s,
            }
            for s, c, cs in scored
        ]

        # Compute new target_rank within the critic-reranked top-N
        target_id = None
        if item.get("target_id"):
            target_id = str(item["target_id"])
        elif item.get("target_path"):
            tp = item["target_path"]
            tp = tp[0] if isinstance(tp, list) else tp
            if tp:
                target_id = get_stem(str(tp))

        new_rank = None
        for i, (_, c, _) in enumerate(scored, start=1):
            cid = c.get("image_id") or get_stem(c.get("image_name", ""))
            if cid == target_id:
                new_rank = i
                break
        new_item["target_result_critic"] = {
            "target_id": target_id,
            "target_rank": new_rank if new_rank is not None else f">{args.topn}",
        }

        out.append(new_item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    cache_fp.close()

    print(f"[Done] saved: {args.output}")
    print(f"[Done] critic calls: {n_calls} new + {n_cached} cached  (total {n_calls + n_cached})")


if __name__ == "__main__":
    main()
