import json
import torch
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.sample_rate = 16000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        MODEL_PRETRAIN = "facebook/wav2vec2-base-960h"

        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_PRETRAIN)
        self.model = Wav2Vec2Model.from_pretrained(MODEL_PRETRAIN).to(self.device)

        model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(model_config, "output")
        self.output_type = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        responses = []
        print(len(requests))
        for request in requests:
            waveform = pb_utils.get_input_tensor_by_name(request, "waveform")
            waveform = waveform.as_numpy()
            waveform = torch.tensor(waveform, dtype=torch.float32)

            input_data = self.process_audio(waveform)

            with torch.no_grad():
                output_data = self.model(input_data).last_hidden_state

            output_data = np.array(output_data.cpu(), dtype=np.float32)

            output = pb_utils.Tensor("output", output_data.astype(self.output_type))

            inference_response = pb_utils.InferenceResponse([output])
            responses.append(inference_response)

        return responses
    
    def process_audio(self, waveform):
        input_values = self.processor(
            waveform.squeeze().numpy(),
            return_tensors="pt",
            sampling_rate=self.sample_rate
        ).input_values

        return input_values.to(self.device)
    
    def finalize(self):
        print('Cleaning up...')
