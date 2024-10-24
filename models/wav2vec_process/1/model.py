import time
import json
import torch
import requests
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import Wav2Vec2Processor
import torchaudio


class TritonPythonModel:
    def initialize(self, args):
        self.sample_rate = 16000
        self.TRITON_SERVER_URL = "http://localhost:8000/v2/models/wav2vec_vino/infer"

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(model_config, "output")
        self.output_type = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            waveform = pb_utils.get_input_tensor_by_name(request, "waveform")
            sample_rate = pb_utils.get_input_tensor_by_name(request, "sample_rate")

            sample_rate = int(sample_rate.as_numpy().item())
            waveform = waveform.as_numpy()
            waveform = torch.tensor(waveform, dtype=torch.float32)

            input_data = self.process_audio(waveform, sample_rate)

            model_response = self.call_model(input_data)

            output_data = model_response["outputs"] 

            output_data = np.array(output_data[0]['data'], dtype=np.float32)

            output = pb_utils.Tensor("output", output_data.astype(self.output_type))

            inference_response = pb_utils.InferenceResponse([output])
            responses.append(inference_response)

        return responses 
    
    def call_model(self, input_values):
        input_data = input_values.numpy().astype('float32').tolist()

        data = {
            "inputs": [
                {
                    "name": "input",
                    "shape": input_values.shape,
                    "datatype": "FP32",
                    "data": input_data
                }
            ]
        }

        headers = {"Content-Type": "application/json"}

        # time.sleep(0.5)

        response = requests.post(self.TRITON_SERVER_URL, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return response.raise_for_status()
    
    def process_audio(self, waveform, sample_rate):
        # if sample_rate != self.sample_rate:
        #     resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
        #     waveform = resample_transform(waveform)

        input_values = self.processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=self.sample_rate).input_values

        return input_values
    
    def finalize(self):
        print('Cleaning up...')