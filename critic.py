# -*- coding: utf-8 -*-
"""
Unified VLM Critic for CIR reranking.
"""
import json
import re
import sys

import torch
from PIL import Image

from prompts import CRITIC_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON found in model output: {text!r}")
    raw_json = m.group()
    depth = 0
    end_idx = 0
    for i, ch in enumerate(raw_json):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    raw_json = raw_json[:end_idx]
    obj = json.loads(raw_json)
    edit = max(0, min(10, int(obj.get("edit", 0))))
    preserve = max(0, min(10, int(obj.get("preserve", 0))))
    reason = str(obj.get("reason", ""))[:300]
    return {"edit": edit, "preserve": preserve, "reason": reason, "raw": text.strip()}


class VLMCritic:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        device_map: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 128,
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)

        model_lower = model_id.lower()
        if "qwen" in model_lower and "llava" not in model_lower and "internvl" not in model_lower:
            self.model_type = "qwen"
            self._init_qwen(model_id, device_map, dtype)
        elif "onevision" in model_lower:
            self.model_type = "llava_ov"
            self._init_llava_onevision(model_id, device_map, dtype)
        elif "llava" in model_lower:
            self.model_type = "llava"
            self._init_llava(model_id, device_map, dtype)
        elif "internvl" in model_lower:
            self.model_type = "internvl"
            self._init_internvl(model_id, device_map, dtype)
        elif "molmo" in model_lower:
            self.model_type = "molmo"
            self._init_molmo(model_id, device_map, dtype)
        else:
            raise ValueError(f"Unsupported model: {model_id}")

        print(f"[VLMCritic] Ready: {model_id} on {device_map}")

    # ── Qwen ──
    def _init_qwen(self, model_id, device_map, dtype):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        try:
            from qwen_vl_utils import process_vision_info
            self._qwen_process_vision = process_vision_info
        except ImportError:
            self._qwen_process_vision = None

        print(f"[VLMCritic] Loading Qwen: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map
        ).eval()

    def _score_qwen(self, ref, cand, prompt_text):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "REFERENCE image:"},
                {"type": "image", "image": ref},
                {"type": "text", "text": "CANDIDATE image:"},
                {"type": "image", "image": cand},
                {"type": "text", "text": prompt_text},
            ],
        }]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self._qwen_process_vision:
            image_inputs, video_inputs = self._qwen_process_vision(messages)
        else:
            image_inputs, video_inputs = [ref, cand], None

        inputs = self.processor(
            text=[chat_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        out_ids = self.model.generate(**inputs, **self.gen_kwargs)
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # ── LLaVA-NeXT ──
    def _init_llava(self, model_id, device_map, dtype):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        print(f"[VLMCritic] Loading LLaVA-NeXT: {model_id}")
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map
        ).eval()

    def _score_llava(self, ref, cand, prompt_text):
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": f"REFERENCE image is the first image.\nCANDIDATE image is the second image.\n\n{prompt_text}"},
            ],
        }]
        chat_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=chat_text, images=[ref, cand], return_tensors="pt"
        ).to(self.model.device)

        out_ids = self.model.generate(**inputs, **self.gen_kwargs)
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # ── LLaVA-OneVision ──
    def _init_llava_onevision(self, model_id, device_map, dtype):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        print(f"[VLMCritic] Loading LLaVA-OneVision: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map
        ).eval()

    def _score_llava_onevision(self, ref, cand, prompt_text):
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": f"REFERENCE image is the first image.\nCANDIDATE image is the second image.\n\n{prompt_text}"},
            ],
        }]
        chat_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=chat_text, images=[ref, cand], return_tensors="pt"
        ).to(self.model.device)

        out_ids = self.model.generate(**inputs, **self.gen_kwargs)
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # ── InternVL2.5 / InternVL3 ──
    def _init_internvl(self, model_id, device_map, dtype):
        print(f"[VLMCritic] Loading InternVL: {model_id}")

        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model_id)

        import os
        if os.path.exists(os.path.join(model_path, "tokenization_internlm2.py")):
            sys.path.insert(0, model_path)
            from tokenization_internlm2 import InternLM2Tokenizer
            self.tokenizer = InternLM2Tokenizer(vocab_file=model_path + "/tokenizer.model")
            import json as _json
            with open(model_path + "/tokenizer_config.json") as _f:
                _tcfg = _json.load(_f)
            _sp = [v["content"] for v in _tcfg.get("added_tokens_decoder", {}).values() if v.get("special")]
            if _sp:
                self.tokenizer.add_tokens(_sp, special_tokens=True)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        from transformers import AutoModel
        import transformers.modeling_utils as _mu
        _need_patch = hasattr(_mu.PreTrainedModel, "_move_missing_keys_from_meta_to_device")
        if _need_patch:
            _orig_move = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
            def _patched_move(self_inner, *a, **kw):
                if not hasattr(self_inner, "all_tied_weights_keys"):
                    self_inner.all_tied_weights_keys = {}
                return _orig_move(self_inner, *a, **kw)
            _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(device_map)
        if _need_patch:
            _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _orig_move

    def _score_internvl(self, ref, cand, prompt_text):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        ref_t = transform(ref).unsqueeze(0)
        cand_t = transform(cand).unsqueeze(0)
        pixel_values = torch.cat([ref_t, cand_t], dim=0).to(
            self.model.device, dtype=torch.bfloat16
        )

        num_patches_list = [1, 1]

        question = f"Image-1: <image>\nImage-2: <image>\nREFERENCE image is Image-1.\nCANDIDATE image is Image-2.\n\n{prompt_text}"

        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        response = self.model.chat(
            self.tokenizer, pixel_values, question, generation_config,
            num_patches_list=num_patches_list
        )
        return response

    # ── Molmo ──
    def _init_molmo(self, model_id, device_map, dtype):
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"[VLMCritic] Loading Molmo: {model_id}")
        self._molmo_dtype = dtype
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True,
            device_map=device_map
        ).eval()

    def _score_molmo(self, ref, cand, prompt_text):
        from transformers import GenerationConfig
        text = f"REFERENCE image is the first image.\nCANDIDATE image is the second image.\n\n{prompt_text}"
        inputs = self.processor.process(images=[ref, cand], text=text)
        inputs_batch = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.model.device)
                if v.dtype == torch.float32:
                    v = v.to(self._molmo_dtype)
                v = v.unsqueeze(0)
            inputs_batch[k] = v

        gen_cfg = GenerationConfig(max_new_tokens=self.max_new_tokens, do_sample=False)
        out_ids = self.model.generate_from_batch(inputs_batch, generation_config=gen_cfg)
        gen_ids = out_ids[0, inputs_batch["input_ids"].size(1):]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # ── Common score ──
    @torch.no_grad()
    def score(self, ref_img_path: str, mod_text: str, cand_img_path: str) -> dict:
        ref = Image.open(ref_img_path).convert("RGB")
        cand = Image.open(cand_img_path).convert("RGB")
        prompt_text = CRITIC_PROMPT.format(mod_text=mod_text.replace(chr(34), chr(39)))

        if self.model_type == "qwen":
            text = self._score_qwen(ref, cand, prompt_text)
        elif self.model_type == "llava":
            text = self._score_llava(ref, cand, prompt_text)
        elif self.model_type == "llava_ov":
            text = self._score_llava_onevision(ref, cand, prompt_text)
        elif self.model_type == "internvl":
            text = self._score_internvl(ref, cand, prompt_text)
        elif self.model_type == "molmo":
            text = self._score_molmo(ref, cand, prompt_text)

        return _parse_json(text)
