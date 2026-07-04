import onnx

model = onnx.load("backbone_shaped.onnx")
for node in model.graph.node:
    if node.op_type in ("Conv", "MaxPool", "AveragePool"):
        attrs = {a.name: list(a.ints) if a.ints else a.i for a in node.attribute}
        print(node.op_type, node.name, "->", attrs)