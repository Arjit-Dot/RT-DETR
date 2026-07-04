import argparse
import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

def preprocess(image_path, target_size=(640, 640)):
    # Load image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        return None, None
    
    h, w = orig_img.shape[:2]
    
    # Resize to model input size
    blob = cv2.resize(orig_img, target_size)
    # Convert BGR to RGB
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1] and transpose to CHW
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    # Add batch dimension (1, 3, 640, 640)
    blob = np.expand_dims(blob, axis=0)
    
    return blob, (h, w)

def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    print("Initializing ONNX Runtime sessions...")
    bb_session = ort.InferenceSession(os.path.join(args.onnx_dir, "backbone.onnx"))
    head_session = ort.InferenceSession(os.path.join(args.onnx_dir, "head.onnx"))
    
    # Get all image paths
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
        
    print(f"Found {len(image_paths)} images to process.")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        input_blob, orig_shape = preprocess(img_path)
        if input_blob is None:
            continue
            
        # 1. Run Backbone
        bb_inputs = {bb_session.get_inputs()[0].name: input_blob}
        onnx_feats = bb_session.run(None, bb_inputs)
        
        # 2. Run Head
        # Pass the original dimensions as float32 to match your successful export
        orig_sizes = np.array([[640, 640]], dtype=np.float32) 
        
        head_inputs = {
            "feat0": onnx_feats[0],
            "feat1": onnx_feats[1],
            "feat2": onnx_feats[2],
            "orig_target_sizes": orig_sizes
        }
        labels, boxes, scores = head_session.run(None, head_inputs)
        
        # 3. Draw Results & Save
        orig_img = cv2.imread(img_path)
        h_orig, w_orig = orig_shape
        
        # Unpack predictions (Batch size is 1)
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]
        
        count = 0
        for label, box, score in zip(labels, boxes, scores):
            if score < args.threshold:
                continue
                
            count += 1
            # Scale coordinates from the 640x640 space back to original image size
            x1 = int(box[0] * (w_orig / 640.0))
            y1 = int(box[1] * (h_orig / 640.0))
            x2 = int(box[2] * (w_orig / 640.0))
            y2 = int(box[3] * (h_orig / 640.0))
            
            # Draw bounding box
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Class {label}: {score:.2f}"
            cv2.putText(orig_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Save output image
        out_path = os.path.join(args.output, filename)
        cv2.imwrite(out_path, orig_img)
        print(f"Processed {filename} -> Found {count} objects -> Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="Path to your folder of custom images")
    parser.add_argument("-d", "--onnx_dir", default="split_onnx", help="Path to folder containing backbone.onnx and head.onnx")
    parser.add_argument("-o", "--output", default="output_predictions", help="Path to save annotated images")
    parser.add_argument("-t", "--threshold", default=0.45, type=float, help="Confidence threshold")
    args = parser.parse_args()
    main(args)