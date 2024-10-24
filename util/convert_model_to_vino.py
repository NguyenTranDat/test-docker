import os
import torch
from openvino.runtime import Core
from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

dummy_input = torch.randn(1, 16000)  

scripted_model = torch.jit.script(model)
torch.save(scripted_model.state_dict(), "model_scripted.pth")

ie = Core()

model_xml = "model.xml"  
model_bin = "model.bin"   

ie.read_model(model_xml, model_bin)
print("Model converted to OpenVINO format.")
