import numpy as np
import requests
from tritonclient.utils import *
import tritonclient.http as httpclient
import torch
import time
import threading
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


def infer(file_path):
    # inputs = prepare_input(waveform, sample_rate)
    
    # response = requests.post(TRITON_SERVER_URL, json=inputs)

    # if response.status_code != 200:
    #     raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # output_data = response.json()
    # print(output_data['outputs'][0]['data'])

    with httpclient.InferenceServerClient("localhost:8000") as client:
        audio = AudioSegment.from_wav(file_path)
        samples = audio.get_array_of_samples()
        waveform = torch.tensor(samples).float().unsqueeze(0)  
        sample_rate = audio.frame_rate

        sample_rate = np.array([[sample_rate]], dtype=np.float32)
        waveform = waveform.numpy()

        inputs = [
            httpclient.InferInput("waveform", waveform.shape, np_to_triton_dtype(waveform.dtype)),
            httpclient.InferInput("sample_rate", sample_rate.shape, np_to_triton_dtype(sample_rate.dtype))
        ]

        inputs[0].set_data_from_numpy(waveform)
        inputs[1].set_data_from_numpy(sample_rate)

        outputs = [
            httpclient.InferRequestedOutput("output"),
        ]

        response = client.infer(MODEL_NAME, inputs, request_id=str(1), outputs=outputs)

        # result = response.get_response()
        output_data = response.as_numpy("output")

        print(output_data)


if __name__ == '__main__':
    file_paths = [
        './data/dia0_utt0.wav',
        './data/dia0_utt1.wav',
        './data/dia0_utt2.wav',
        './data/dia0_utt3.wav'
    ]

    threads = []
    start_time = time.time()

    for file_path in file_paths:
        thread = threading.Thread(target=infer, args=(file_path,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()

    print(f"Total time taken: {end_time - start_time:.2f} seconds")
