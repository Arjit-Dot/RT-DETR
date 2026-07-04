import onnx

print("Loading ONNX model...")
model = onnx.load("backbone_shaped.onnx")

patch_count = 0
for node in model.graph.node:
    if node.op_type in ['AveragePool', 'MaxPool']:
        for attr in node.attribute:
            if attr.name == 'ceil_mode' and attr.i == 1:
                attr.i = 0  # Force ceil_mode to 0
                patch_count += 1
                print(f"Patched ceil_mode in node: {node.name}")

onnx.save(model, "backbone_patched.onnx")
print(f"Successfully patched {patch_count} nodes. Saved as backbone_patched.onnx")
