# pip install torch onnx onnx-tf tensorflow

'''
import torch
import ai_edge_torch  # Ensure this library is installed and available

# Load the Parseq model
model = torch.hub.load('baudm/parseq', 'trba', pretrained=True).eval()

# Save the model's state dictionary
torch.save(model.state_dict(), 'parseq_trba_state_dict.pt')

# Define a sample input with the correct shape
# Adjust the input shape according to the Parseq model's requirements
# Example: (batch_size, channels, height, width)
sample_input = (torch.randn(1, 3, 32, 128),)  # Adjust based on Parseq's expected input

# Convert the PyTorch model to an edge-optimized format
edge_model = ai_edge_torch.convert(model, sample_input)

# Run inference using the edge model
output = edge_model(*sample_input)

# Export the model to TFLite format
edge_model.export('parseq_trba.tflite')

print("TFLite model exported successfully!")
'''

# Re-run your code
import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load the Parseq model
model = torch.hub.load('baudm/parseq', 'trba', pretrained=True).eval()

# Define a dummy input with the correct shape
dummy_input = torch.randn(1, 3, 32, 128)  # Adjust based on the model's requirements

# Export the model to ONNX
onnx_path = "parseq.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

# Load the ONNX model
onnx_model = onnx.load(onnx_path)

# Convert the ONNX model to TensorFlow
tf_rep = prepare(onnx_model)

# Save the TensorFlow model
tf_rep.export_graph("parseq_tf")

# Load the TensorFlow model
tf_model = tf.saved_model.load("parseq_tf")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("parseq_tf")
tflite_model = converter.convert()

# Save the TFLite model
tflite_path = "parseq.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")