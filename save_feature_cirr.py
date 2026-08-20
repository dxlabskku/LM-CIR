# Dependencies: pip install open_clip_torch pillow torch torchvision tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TQDM_ASCII"] = " 123456789#"
import json
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

# Select the OpenCLIP backbone here; its pretrained weights are chosen automatically.
pretraining = {
    "ViT-B-32": "laion2b_s34b_b79k",
    "ViT-L-14": "laion2b_s32b_b82k",
    "ViT-bigG-14": "laion2b_s39b_b160k"
}
MODEL_NAME = "ViT-L-14"

class FeatureBuilder:
    def __init__(
        self,
        model_name=None,
        pretrained=None,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = MODEL_NAME if model_name is None else model_name
        if pretrained is None:
            if model_name not in pretraining:
                raise KeyError(
                    f"model_name={model_name!r} is not in pretraining; add it to the mapping "
                    "or pass pretrained explicitly."
                )
            pretrained = pretraining[model_name]
        self.model_name = model_name
        self.pretrained = pretrained

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device).eval()

    def collect_images_from_split_json(self, json_path, images_base_dir):
        """Collect the image paths listed in a CIRR split file."""
        with open(json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        all_image_paths = []
        for rel in mapping.values():
            full = os.path.normpath(os.path.join(images_base_dir, rel))
            all_image_paths.append(full)

        all_image_paths = sorted(all_image_paths)
        return all_image_paths

    def check_duplicate_names(self, image_paths):
        """Ensure image names are globally unique before saving."""
        name_to_paths = {}
        for path in image_paths:
            name = os.path.basename(path)
            if name not in name_to_paths:
                name_to_paths[name] = []
            name_to_paths[name].append(path)

        duplicates = {k: v for k, v in name_to_paths.items() if len(v) > 1}

        if duplicates:
            print("\n[Error] Found duplicate image names across folders:")
            for name, paths in duplicates.items():
                print(f"\nDuplicate name: {name}")
                for p in paths:
                    print(f"  {p}")
            raise ValueError(
                f"Found {len(duplicates)} duplicated image names. "
                f"Since you want to save only image names, please resolve duplicates first."
            )

        print(f"[Info] Duplicate check passed. Total unique image names: {len(image_paths)}")

    @torch.no_grad()
    def encode_images(self, image_paths, batch_size=256):
        """Return L2-normalized image features as a CPU tensor."""
        all_feats = []

        for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images", ascii=True):
            batch_paths = image_paths[start:start + batch_size]
            batch_tensors = []

            valid_paths = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = self.preprocess(img)
                    batch_tensors.append(img_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"[Warning] Failed to load image: {path}, error: {e}")

            if len(batch_tensors) == 0:
                continue

            batch_tensors = torch.stack(batch_tensors, dim=0).to(self.device)

            img_feats = self.model.encode_image(batch_tensors)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            img_feats = img_feats.cpu()

            all_feats.append(img_feats)

        if len(all_feats) == 0:
            raise RuntimeError("No valid image features were extracted.")

        all_feats = torch.cat(all_feats, dim=0)
        return all_feats

    def build_and_save(
        self,
        split_json_path,
        images_base_dir,
        save_path,
        batch_size=256,
    ):
        print("[Info] Collecting images from split JSON...")
        image_paths = self.collect_images_from_split_json(split_json_path, images_base_dir)
        print(
            f"[Info] JSON size: {len(image_paths)} entries, "
            f"json file on disk: {os.path.getsize(split_json_path)} bytes"
        )

        if len(image_paths) == 0:
            raise RuntimeError("No images found in the split JSON / base directory.")

        print("[Info] Checking duplicate image names...")
        self.check_duplicate_names(image_paths)

        print("[Info] Encoding image features...")
        features = self.encode_images(image_paths, batch_size=batch_size)

        image_names = [os.path.basename(p) for p in image_paths]

        assert features.shape[0] == len(image_names), \
            f"Feature count {features.shape[0]} != image_names count {len(image_names)}"

        features = features.to(torch.float16)

        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        save_data = {
            "features": features,
            "image_names": image_names,
            "image_paths": image_paths,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "feature_dim": features.shape[1],
            "num_images": features.shape[0],
        }

        torch.save(save_data, save_path)

        print(f"[Info] Saved feature file to: {save_path}")
        print(
            f"[Info] .pt size: {os.path.getsize(save_path)} bytes | "
            f"tensor shape: {tuple(features.shape)}"
        )
        print(f"[Info] Feature dtype: {features.dtype}")


if __name__ == "__main__":
    # Run this script from the CIR directory.
    split_json_path = "CIRR/data/cirr/image_splits/split.rc2.test1.json"
    images_base_dir = "CIRR/data/cirr/images_raw"
    save_path = "LM-CIR/image_features/feature_L-14/cirr_test1_feature.pt"

    builder = FeatureBuilder()

    builder.build_and_save(
        split_json_path=split_json_path,
        images_base_dir=images_base_dir,
        save_path=save_path,
        batch_size=256,  # Reduce this value if GPU memory is limited.
    )