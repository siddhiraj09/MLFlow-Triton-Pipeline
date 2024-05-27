import onnx
from onnx import helper

# Load your existing ONNX model
model = onnx.load("model.onnx")

# Get the model's graph
graph = model.graph

# Modify input tensor shape
for input_tensor in graph.input:
    input_tensor.type.tensor_type.shape.dim[0].dim_param = 'batch_size'

# Modify output tensor shape
for output_tensor in graph.output:
    output_tensor.type.tensor_type.shape.dim[0].dim_param = 'batch_size'

# Save the modified model
onnx.save(model, "model_dynamic.onnx")
