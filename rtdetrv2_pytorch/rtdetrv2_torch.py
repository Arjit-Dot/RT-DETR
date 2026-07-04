"""
Run RT-DETRv2 inference on all images in a folder.

Usage:
    cd rtdetrv2_pytorch
    python infer_folder.py \
        -c configs/rtdetrv2/my_inference.yml \
        -r /path/to/checkpoint.pth \
        -i /path/to/image_folder \
        -o /path/to/output_folder \
        --device cpu   # or mps
"""
import argparse
import glob
import os

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torchvision.transforms as T

from src.core import YAMLConfig


class Model(nn.Module):
    def __init__(self, cfg_path, ckpt_path, device):
        super().__init__()
        cfg = YAMLConfig(cfg_path, resume=ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint.get("ema", {}).get("module") if "ema" in checkpoint else checkpoint.get("model", checkpoint)

        cfg.model.load_state_dict(state)

        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.device = device
        self.to(device)

    def forward(self, images, orig_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_sizes)


def main(args):
    device = torch.device(args.device)
    model = Model(args.config, args.resume, device).eval()

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    os.makedirs(args.output, exist_ok=True)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(args.input, e)))

    print(f"Found {len(paths)} images")

    for path in paths:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        im_tensor = transform(im).unsqueeze(0).to(device)
        orig_size = torch.tensor([[w, h]]).to(device)

        with torch.no_grad():
            labels, boxes, scores = model(im_tensor, orig_size)

        draw = ImageDraw.Draw(im)
        keep = scores[0] > args.threshold
        for box, score, label in zip(boxes[0][keep], scores[0][keep], labels[0][keep]):
            box = box.tolist()
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{int(label)}:{score:.2f}", fill="red")

        out_path = os.path.join(args.output, os.path.basename(path))
        im.save(out_path)
        print(f"Saved {out_path} ({int(keep.sum())} detections)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="path to .pth checkpoint")
    parser.add_argument("-i", "--input", required=True, help="folder of images")
    parser.add_argument("-o", "--output", required=True, help="folder to save results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    args = parser.parse_args()
    main(args)