import numpy as np
import concurrent.futures
from tritonclient.utils import *
import tritonclient.http as httpclient
import torch
import time
import os
from pydub import AudioSegment
import pandas as pd


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

        # print(output_data)


if __name__ == '__main__':
    folder_data_path = './data'
    csv_output="./result/triton_python.csv"
    max_workers = 4
    processing_times = []
    file_counts = []

    file_paths = [os.path.join(folder_data_path, file_path) for file_path in os.listdir(folder_data_path) if file_path.endswith('.wav')]

    for i in range(1,33):
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(infer, file_paths[0:i])

        end_time = time.time()

        processing_times.append(end_time-start_time)
        file_counts.append(i) 

    df = pd.DataFrame({
        "File Count": file_counts,
        "Time (seconds)": processing_times,
    })
    df.to_csv(csv_output, index=False)

    print("DONE!")

