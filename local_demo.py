import argparse
import os
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vits,
    build_efficient_sam_vitt,
)

WEIGHT_URLS = {
    "ti": "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt",
    "s": "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip",
}


def download_if_missing(path: str, url: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading missing weight: {url}")
    urllib.request.urlretrieve(url, path)


def build_model(name: str):
    if name == "ti":
        download_if_missing("weights/efficient_sam_vitt.pt", WEIGHT_URLS["ti"])
        return build_efficient_sam_vitt()
    if name == "s":
        checkpoint = "weights/efficient_sam_vits.pt"
        if not os.path.exists(checkpoint):
            download_if_missing("weights/efficient_sam_vits.pt.zip", WEIGHT_URLS["s"])
            with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", "r") as zip_ref:
                zip_ref.extractall("weights")
        return build_efficient_sam_vits()
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Run a local EfficientSAM point-prompt demo.")
    parser.add_argument("--model", choices=["ti", "s"], default="ti")
    parser.add_argument("--image", default="figs/examples/dogs.jpg")
    parser.add_argument("--points", default="580,350;650,350", help="Point prompts as x,y;x,y")
    parser.add_argument("--output-dir", default="local_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model).to(device).eval()
    image_np = np.array(Image.open(args.image).convert("RGB"))
    image_tensor = transforms.ToTensor()(image_np).to(device)

    points = []
    for item in args.points.split(";"):
        x, y = item.split(",")
        points.append([int(x), int(y)])
    input_points = torch.tensor([[points]], device=device)
    input_labels = torch.ones((1, 1, len(points)), dtype=torch.int64, device=device)

    with torch.no_grad():
        predicted_logits, predicted_iou = model(
            image_tensor[None, ...],
            input_points,
            input_labels,
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().numpy()

    masked_image_np = image_np.astype(np.uint8) * mask[:, :, None]
    mask_path = os.path.join(args.output_dir, f"dogs_efficientsam_{args.model}_mask.png")
    overlay_path = os.path.join(args.output_dir, f"dogs_efficientsam_{args.model}_masked.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    Image.fromarray(masked_image_np).save(overlay_path)
    print(f"device={device}")
    print(f"mask={mask_path}")
    print(f"masked_image={overlay_path}")


if __name__ == "__main__":
    main()
