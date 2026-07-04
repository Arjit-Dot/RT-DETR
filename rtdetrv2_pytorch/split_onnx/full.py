import onnx
from onnx import shape_inference

model = onnx.load("backbone_shaped.onnx")
inferred = shape_inference.infer_shapes(model, strict_mode=True)
onnx.checker.check_model(inferred)
onnx.save(inferred, "backbone_final.onnx")
print("OK — shape inference and checker both passed")