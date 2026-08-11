import os
import json
import base64
import mimetypes
from typing import List, Dict
from prompts import COT_PROMPT_TITLE_ONLY_GENERAL

def local_image_to_data_url(image: str) -> str:
    if not os.path.exists(image):
        raise FileNotFoundError(f"Image not found: {image}")

    mime_type, _ = mimetypes.guess_type(image)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image, "rb") as f:
        img_bytes = f.read()

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{img_base64}"
    return data_url


def build_user_prompt(instruction: str) -> str:
    return f"""
<Input>
{{
    "Original Image": <image_url>,
    "Manipulation text": "{instruction}"
}}
"""


def build_batch_request(
    custom_id: str,
    image_path: str,
    instruction: str,
    sys_prompt: str,
    engine: str = "gpt-5.1-2025-11-13",
    reasoning_effort: str = "low",
    max_output_tokens: int = 200,
) -> Dict:
    data_url = local_image_to_data_url(image_path)
    user_prompt = build_user_prompt(instruction)

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": engine,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": sys_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {
                            "type": "input_image",
                            "image_url": data_url,
                        },
                    ],
                },
            ],
        },
    }


def write_jsonl(records: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chunk_records(records: List[Dict], chunk_size: int) -> List[List[Dict]]:
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


if __name__ == "__main__":
    input_json = "CIRR/data/cirr/captions_ext/cap.ext.rc2.test1.json"
    image_folder = "CIRR/data/cirr/images_raw/test1"
    output_dir = "/output"

    records_per_file = 100

    engine = "gpt-5.1-2025-11-13"

    sys_prompt = COT_PROMPT_TITLE_ONLY_GENERAL
    max_output_tokens = 200

    os.makedirs(output_dir, exist_ok=True)
    with open(input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    records = []
    skipped = 0
    for item in samples:
        custom_id = str(item["pairid"])
        reference = item["reference"]
        instruction = item["caption"]
        image_path = os.path.join(image_folder, f"{reference}.png")

        try:
            record = build_batch_request(
                custom_id=custom_id,
                image_path=image_path,
                instruction=instruction,
                sys_prompt=sys_prompt,
                engine=engine,
                reasoning_effort="low",
                max_output_tokens=max_output_tokens,
            )
            records.append(record)
        except FileNotFoundError:
            skipped += 1

    chunks = chunk_records(records, records_per_file)
    print(f"Total valid requests: {len(records)}")
    print(f"Skipped (image missing): {skipped}")
    print(f"Will write {len(chunks)} jsonl files to: {output_dir}")

    for idx, chunk in enumerate(chunks, 1):
        output_jsonl = os.path.join(output_dir, f"cirr_batch_{idx:04d}.jsonl")
        write_jsonl(chunk, output_jsonl)
        size_bytes = os.path.getsize(output_jsonl)
        size_mb = size_bytes / (1024 * 1024)
        print(
            f"[{idx:04d}] {output_jsonl} | records={len(chunk)} | size={size_bytes} bytes ({size_mb:.2f} MB)"
        )