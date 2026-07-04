import onnx

model = onnx.load("backbone_final.onnx")
for i, node in enumerate(model.graph.node):
    if node.op_type in ("Conv", "MaxPool", "AveragePool"):
        attrs = {a.name: (list(a.ints) if a.ints else a.i) for a in node.attribute}
        print(i, node.op_type, node.name, "->", attrs)