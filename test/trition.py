import requests
import time
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from transformers import Wav2Vec2Processor

TRITON_SERVER_URL = "http://localhost:8000/v2/models/wav2vec2_model/infer"

def preprocess_audio(file_path: str):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio = AudioSegment.from_wav(file_path)

    samples = audio.get_array_of_samples()
    waveform = torch.tensor(samples).float().unsqueeze(0)  
    sample_rate = audio.frame_rate

    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)

    waveform = waveform[:, :16000]

    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    
    return input_values

def send_batch_to_triton(file_paths):
    start_time = time.time()
    
    batch_inputs = []
    for file_path in file_paths:
        batch_inputs.append(preprocess_audio(file_path))

    batch_inputs = torch.cat(batch_inputs, dim=0).numpy().astype(np.float32)

    data = {
        "inputs": [
            {
                "name": "input",
                "shape": batch_inputs.shape,
                "datatype": "FP32",
                "data": batch_inputs.tolist()
            }
        ]
    }

    response = requests.post(TRITON_SERVER_URL, json=data)

    end_time = time.time()
    if response.status_code == 200:
        result = response.json()
        print("Batch Inference Result: ", result)
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print(f"Total time for batch execution: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    file_paths = [
        '/home/trandat/test_docker/data/dia0_utt0.wav',
        '/home/trandat/test_docker/data/dia0_utt1.wav',
        '/home/trandat/test_docker/data/dia0_utt2.wav',
        '/home/trandat/test_docker/data/dia0_utt3.wav'
    ]
    
    send_batch_to_triton(file_paths)
