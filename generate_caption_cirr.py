import os
import json
import time
import base64
import mimetypes
from typing import Set
from openai import OpenAI
from prompts import COT_PROMPT_TITLE_ONLY_GENERAL

# Run this script from the CIR project root.
INPUT_JSON = "CIRR/data/cirr/captions/cap.rc2.test1.json"
IMAGE_FOLDER = "CIRR/data/cirr/images_raw/test1"
OUTPUT_DIR = "LM-CIR/result"

# JSONL stores raw responses for resume/debugging; JSON stores clean records.
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "cirr_direct_results.jsonl")
FORMATTED_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "cirr_direct_merged.json")
FAILED_LOG = os.path.join(OUTPUT_DIR, "cirr_direct_failed.log")
ENGINE = "gpt-5.1-2025-11-13"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 200
REQUEST_INTERVAL = 20
MAX_ATTEMPTS = 3

# OPENAI_API_KEY must be set in the environment.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)

def local_image_to_data_url(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{img_base64}"

def build_user_prompt(instruction: str) -> str:
    return f"""
<Input>
{{
    "Original Image": <image_url>,
    "Manipulation text": "{instruction}"
}}
"""

def call_gpt(image_path: str, instruction: str):
    data_url = local_image_to_data_url(image_path)
    user_prompt = build_user_prompt(instruction)
    # Apply the same delay to initial requests and retries.
    print(f"[WAIT] sleep {REQUEST_INTERVAL}s before API request...")
    time.sleep(REQUEST_INTERVAL)
    response = client.responses.create(
        model=ENGINE,
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": COT_PROMPT_TITLE_ONLY_GENERAL,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
    )
    return response

def convert_to_batch_like_record(custom_id: str, response):
    response_body = json.loads(response.model_dump_json())
    request_id = getattr(response, "_request_id", None)
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": request_id,
            "body": response_body,
        },
        "error": None,
    }

def extract_edited_caption(record: dict):
    try:
        outputs = record["response"]["body"]["output"]
        for output in outputs:
            if output.get("type") != "message":
                continue
            for content in output.get("content", []):
                if content.get("type") != "output_text":
                    continue
                text = content.get("text", "").strip()
                if not text:
                    continue
                payload = json.loads(text)
                edited_caption = payload.get("Target Image Description")
                if edited_caption:
                    return edited_caption
    except Exception as e:
        print(f"[WARN] extraction failed custom_id={record.get('custom_id')}, error={e}")
    return None

def build_formatted_record(item: dict, edited_caption: str) -> dict:
    reference_id = item.get("reference", "")
    # test1 has no target_hard, so target fields remain empty.
    target_id = item.get("target_hard", "") or ""
    return {
        "pairid": int(item["pairid"]),
        "reference_id": reference_id,
        "target_id": target_id,
        "reference_path": os.path.join(IMAGE_FOLDER, f"{reference_id}.png"),
        "target_path": (
            os.path.join(IMAGE_FOLDER, f"{target_id}.png") if target_id else ""
        ),
        "instruction": item.get("caption", ""),
        "edited_caption": edited_caption,
    }

def load_formatted_records(output_path: str) -> dict:
    if not os.path.exists(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Formatted output must be a JSON list: {output_path}")
    return {
        str(record["pairid"]): record
        for record in records
        if isinstance(record, dict) and "pairid" in record
    }

def save_formatted_records(
    records_by_id: dict,
    caption_data: list,
    output_path: str,
):
    records = [
        records_by_id[str(item["pairid"])]
        for item in caption_data
        if str(item["pairid"]) in records_by_id
    ]
    # Replace atomically to avoid leaving a partial JSON file.
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, output_path)

def append_jsonl(record: dict, output_path: str):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def append_failed(
    custom_id: str,
    reference: str,
    instruction: str,
    error: Exception,
):
    record = {
        "custom_id": custom_id,
        "reference": reference,
        "instruction": instruction,
        "error": repr(error),
    }
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_completed_ids(output_path: str) -> Set[str]:
    # Read successful IDs from JSONL so interrupted runs can resume.
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] invalid JSON line={line_num}")
                continue
            custom_id = record.get("custom_id")
            response = record.get("response")
            if (
                custom_id is not None
                and response is not None
                and response.get("status_code") == 200
            ):
                completed.add(str(custom_id))
    return completed

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading input data...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)
    # Debug limit: remove this line to process the full dataset.
    samples = samples[:3]
    print(f"Total samples: {len(samples)}")
    completed_ids = load_completed_ids(OUTPUT_JSONL)
    print(f"Already completed: {len(completed_ids)}")
    print(f"Output file: {OUTPUT_JSONL}")
    formatted_records = load_formatted_records(FORMATTED_OUTPUT_JSON)
    print(f"Formatted output: {FORMATTED_OUTPUT_JSON}")
    success_count = 0
    failed_count = 0
    skipped_count = 0
    missing_image_count = 0
    for index, item in enumerate(samples, 1):
        custom_id = str(item["pairid"])
        reference = item["reference"]
        instruction = item["caption"]
        if custom_id in completed_ids:
            skipped_count += 1
            print(
                f"[{index}/{len(samples)}] pairid={custom_id} "
                "SKIP (already completed)"
            )
            continue
        image_path = os.path.join(IMAGE_FOLDER, f"{reference}.png")
        if not os.path.exists(image_path):
            missing_image_count += 1
            print(
                f"[{index}/{len(samples)}] pairid={custom_id} "
                f"MISSING IMAGE: {image_path}"
            )
            continue
        print(f"\n[{index}/{len(samples)}] pairid={custom_id}")
        print(f"reference={reference}")
        print(f"instruction={instruction}")
        success = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                print(f"[API] attempt {attempt}/{MAX_ATTEMPTS}")
                response = call_gpt(image_path=image_path, instruction=instruction)
                print("[GPT OUTPUT]")
                print(response.output_text)
                record = convert_to_batch_like_record(
                    custom_id=custom_id,
                    response=response,
                )
                edited_caption = extract_edited_caption(record)
                if edited_caption is None:
                    raise ValueError("GPT output lacks Target Image Description")
                formatted_records[custom_id] = build_formatted_record(
                    item,
                    edited_caption,
                )
                save_formatted_records(
                    formatted_records,
                    samples,
                    FORMATTED_OUTPUT_JSON,
                )
                append_jsonl(record, OUTPUT_JSONL)
                completed_ids.add(custom_id)
                success_count += 1
                success = True
                print(f"[SUCCESS] pairid={custom_id}")
                break
            except Exception as e:
                print(f"[ERROR] pairid={custom_id}, attempt={attempt}")
                print(repr(e))
                if attempt < MAX_ATTEMPTS:
                    print("[RETRY] will retry...")
                else:
                    failed_count += 1
                    append_failed(
                        custom_id=custom_id,
                        reference=reference,
                        instruction=instruction,
                        error=e,
                    )
                    print("[FAILED] maximum attempts reached")
        print("--------------------------------")
        print(
            f"success={success_count}, failed={failed_count}, "
            f"skipped={skipped_count}, missing={missing_image_count}"
        )
    print("\n==============================")
    print("Finished")
    print("==============================")
    print(f"success: {success_count}")
    print(f"failed: {failed_count}")
    print(f"skipped existing: {skipped_count}")
    print(f"missing image: {missing_image_count}")
    print(f"raw results saved to: {OUTPUT_JSONL}")
    print(f"formatted results saved to: {FORMATTED_OUTPUT_JSON}")

if __name__ == "__main__":
    main()
