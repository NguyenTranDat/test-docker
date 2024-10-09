import json
import torch
import numpy as np
from pydub import AudioSegment


file_path = './data/dia0_utt0.wav'

def preprocess(file_path: str):
    audio = AudioSegment.from_wav(file_path)
    samples = audio.get_array_of_samples()
    waveform = torch.tensor(samples).float().unsqueeze(0)  
    sample_rate = audio.frame_rate

    return waveform, sample_rate


def create_json(file_path: str):
    waveform, sample_rate = preprocess(file_path)
    waveform_np = np.array(waveform, dtype=np.float32)
    sample_rate_np = np.array([sample_rate], dtype=np.float32)

    data = {
        "data": [
                {
                    "waveform": waveform_np.flatten().tolist(),
                    "sample_rate": sample_rate_np.tolist(),
                },
        ]
    }
    with open('./perf_analyzer_test/test.json', 'w') as f:
        json.dump(data, f)


create_json(file_path)