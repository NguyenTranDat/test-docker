import tritonclient.http as httpclient
import numpy as np

triton_client = httpclient.InferenceServerClient(url="localhost:8000")

with open("anh.jpg", "rb") as f:
    image_data = f.read()

image_array = np.frombuffer(image_data, dtype=np.uint8)

batch_size = 1

inputs = [
    httpclient.InferInput("DALI_INPUT_0", [batch_size, image_array.shape[0]], "UINT8")
]

inputs[0].set_data_from_numpy(np.expand_dims(image_array, axis=0))

outputs = [
    httpclient.InferRequestedOutput("DALI_OUTPUT_0")
]

response = triton_client.infer(model_name="dali", inputs=inputs, outputs=outputs)

output_data = response.as_numpy("DALI_OUTPUT_0")

print("Output shape:", output_data.shape)
