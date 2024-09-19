import torch
from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

model.eval()

dummy_input = torch.randn(1, 16000)

torch.onnx.export(
    model,
    dummy_input,
    "/home/trandat/test_docker/models/wav2vec2_model/1/model.onnx",
    input_names=["input"], 
    output_names=["output"],
    export_params=True,
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("Model exported to ONNX format.")
