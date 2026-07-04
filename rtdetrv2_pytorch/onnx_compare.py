import argparse
import torch
import numpy as np
import onnxruntime as ort
from src.core import YAMLConfig

def main(args):
    print("Loading PyTorch baseline...")
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint.get("ema", {}).get("module") if "ema" in checkpoint else checkpoint.get("model", checkpoint)
    cfg.model.load_state_dict(state)
    
    full_model = cfg.model.deploy().eval()
    postprocessor = cfg.postprocessor.deploy().eval()

    print(f"Loading ONNX models from {args.onnx_dir}/...")
    # Initialize ONNX Runtime sessions
    backbone_session = ort.InferenceSession(f"{args.onnx_dir}/backbone.onnx")
    head_session = ort.InferenceSession(f"{args.onnx_dir}/head.onnx")

    # Generate identical dummy inputs for both pipelines
    dummy_img = torch.rand(1, 3, 640, 640)
    dummy_sizes = torch.tensor([[640, 640]], dtype=torch.float32)

    print("Running PyTorch pipeline...")
    with torch.no_grad():
        # Replicate the exact flow of the full model
        pt_feats = full_model.backbone(dummy_img)
        pt_encoded = full_model.encoder(pt_feats)
        pt_decoded = full_model.decoder(pt_encoded)
        pt_out = postprocessor(pt_decoded, dummy_sizes)
        
        # Unpack depending on how your specific postprocessor returns data
        if isinstance(pt_out, dict):
            pt_labels, pt_boxes, pt_scores = pt_out["labels"], pt_out["boxes"], pt_out["scores"]
        else:
            pt_labels, pt_boxes, pt_scores = pt_out[0], pt_out[1], pt_out[2]

    print("Running ONNX pipeline...")
    # 1. Run ONNX Backbone
    bb_inputs = {backbone_session.get_inputs()[0].name: dummy_img.numpy()}
    onnx_feats = backbone_session.run(None, bb_inputs)
    
    # 2. Run ONNX Head (connecting backbone outputs to head inputs)
    head_inputs = {
        "feat0": onnx_feats[0],
        "feat1": onnx_feats[1],
        "feat2": onnx_feats[2],
        "orig_target_sizes": dummy_sizes.numpy()
    }
    onnx_labels, onnx_boxes, onnx_scores = head_session.run(None, head_inputs)

    print("\n--- 📊 Comparison Results ---")
    
    # Calculate the maximum absolute difference
    score_diff = np.abs(pt_scores.numpy() - onnx_scores).max()
    box_diff = np.abs(pt_boxes.numpy() - onnx_boxes).max()
    label_diff = np.abs(pt_labels.numpy() - onnx_labels).max()

    print(f"Max Score Difference: {score_diff:.8f}")
    print(f"Max Box Difference:   {box_diff:.8f}")
    print(f"Max Label Difference: {label_diff:.8f}")

    # Standard acceptable tolerance for float32 ONNX exports is usually < 1e-4
    if max(score_diff, box_diff) < 1e-4:
        print("\n✅ SUCCESS: The split ONNX models match the PyTorch baseline perfectly!")
    else:
        print("\n⚠️ WARNING: Significant difference detected! The ONNX export may have altered the math.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-d", "--onnx_dir", default="split_onnx")
    args = parser.parse_args()
    main(args)