import numpy as np
import tritonclient.http as httpclient

# Define the server URL and model name
url = "localhost:8000"
model_name = "mnist_model"

# Create a Triton HTTP client
client = httpclient.InferenceServerClient(url=url)

# Prepare input data
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

# Create the input object
inputs = httpclient.InferInput("onnx::Flatten_0", input_data.shape, "FP32")
inputs.set_data_from_numpy(input_data)

# Create the output object
outputs = httpclient.InferRequestedOutput("8")

# Send the request to the server
response = client.infer(model_name, inputs=[inputs], outputs=[outputs])

# Get the results
result = response.as_numpy("8")

# Print the result
print("Inference result:", result)
