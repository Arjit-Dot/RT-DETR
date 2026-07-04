"""
Split RT-DETRv2 into two ONNX graphs:
  1. backbone.onnx   -> convert with Cubie ACUITY SDK / TIM-VX and run on the NPU
  2. head.onnx        -> run on CPU with onnxruntime (hybrid encoder + decoder + postprocess)

Run from inside rtdetrv2_pytorch/ so `src` is importable.

Usage:
    python split_export.py -c configs/rtdetrv2/my_inference.yml -r checkpoint.pth -o out/
"""
import argparse
import os

import torch
import torch.nn as nn

from src.core import YAMLConfig


class BackboneOnly(nn.Module):
    """Just the conv backbone -> goes to the NPU."""
    def __init__(self, full_model):
        super().__init__()
        self.backbone = full_model.backbone

    def forward(self, images):
        # Returns the list/tuple of multi-scale feature maps (e.g. [C3, C4, C5])
        return self.backbone(images)


class HeadOnly(nn.Module):
    """Hybrid encoder + decoder + postprocessor -> stays on CPU."""
    def __init__(self, full_model, postprocessor):
        super().__init__()
        self.encoder = full_model.encoder
        self.decoder = full_model.decoder
        self.postprocessor = postprocessor

    def forward(self, feat0, feat1, feat2, orig_target_sizes):
        feats = [feat0, feat1, feat2]
        encoded = self.encoder(feats)
        outputs = self.decoder(encoded)
        return self.postprocessor(outputs, orig_target_sizes)


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    print("Printing what claude asked")
    #print(cfg.model)
    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint.get("ema", {}).get("module") if "ema" in checkpoint else checkpoint.get("model", checkpoint)
    cfg.model.load_state_dict(state)

    full_model = cfg.model.deploy().eval()
    postprocessor = cfg.postprocessor.deploy().eval()

    os.makedirs(args.output, exist_ok=True)

    # --- 1. Export backbone (NPU-bound) ---
    backbone_wrap = BackboneOnly(full_model).eval()
    dummy_img = torch.rand(1, 3, 640, 640)

    with torch.no_grad():
        feats = backbone_wrap(dummy_img)
    print(f"Backbone produced {len(feats)} feature maps, shapes: {[f.shape for f in feats]}")

    torch.onnx.export(
        backbone_wrap,
        dummy_img,
        os.path.join(args.output, "backbone.onnx"),
        input_names=["images"],
        output_names=[f"feat{i}" for i in range(len(feats))],
        opset_version=11,
        do_constant_folding=True,
    )
    print("Wrote backbone.onnx  <- convert this one with Acuity SDK / TIM-VX for the NPU")

    # --- 2. Export head (CPU-bound: hybrid encoder + decoder + postprocess) ---
    head_wrap = HeadOnly(full_model, postprocessor).eval()
    dummy_sizes = torch.tensor([[640, 640]], dtype=torch.float32)

    torch.onnx.export(
        head_wrap,
        (*feats, dummy_sizes),
        os.path.join(args.output, "head.onnx"),
        input_names=[f"feat{i}" for i in range(len(feats))] + ["orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        opset_version=17,
        do_constant_folding=True,
    )
    print("Wrote head.onnx  <- run this one with onnxruntime on the ARM CPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-o", "--output", default="split_onnx")
    args = parser.parse_args()
    main(args)