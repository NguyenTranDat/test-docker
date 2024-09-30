import numpy as np
import requests
import torch
from pydub import AudioSegment


MODEL_NAME = 'wav2vec_py'
TRITON_SERVER_URL = f'http://localhost:8000/v2/models/{MODEL_NAME}/infer'


def prepare_input(waveform, sample_rate):
    waveform_np = np.array(waveform, dtype=np.float32)
    sample_rate_np = np.array([sample_rate], dtype=np.float32)

    inputs = {
        "inputs": [
            {
                "name": "waveform",
                "shape": waveform_np.shape,
                "datatype": "FP32",
                "data": waveform_np.flatten().tolist()
            },
            {
                "name": "sample_rate",
                "shape": sample_rate_np.shape,
                "datatype": "FP32",
                "data": sample_rate_np.tolist()
            }
        ]
    }
    
    return inputs


def infer(waveform, sample_rate):
    inputs = prepare_input(waveform, sample_rate)
    
    response = requests.post(TRITON_SERVER_URL, json=inputs)

    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    output_data = response.json()
    return output_data['outputs'][0]['data']


def preprocess_audio(file_path: str):
    audio = AudioSegment.from_wav(file_path)
    samples = audio.get_array_of_samples()
    waveform = torch.tensor(samples).float().unsqueeze(0)  
    sample_rate = audio.frame_rate

    return waveform, sample_rate


if __name__ == '__main__':
    file_paths = [
        '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt0.wav',
        '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt1.wav',
        '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt2.wav',
        '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt3.wav'
    ]

    for file_path in file_paths:
        waveform_example, sample_rate = preprocess_audio(file_path)
        result = infer(waveform_example, [sample_rate])

    print("Kết quả:", result)
