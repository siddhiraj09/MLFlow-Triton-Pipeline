import onnx

# Load the ONNX model
model = onnx.load("model.onnx")

# Print the model's input names
for input in model.graph.input:
    print(input.name)

# Print the model's output names
for output in model.graph.output:
    print(output.name)
