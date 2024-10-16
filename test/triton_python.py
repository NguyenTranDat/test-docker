import numpy as np
import threading
import time
import os
from pydub import AudioSegment
import pandas as pd
import torch
import tritonclient.http as httpclient
from tritonclient.utils import *


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

        output_data = response.as_numpy("output")

        del output_data, audio, samples, waveform, response
        torch.cuda.empty_cache()


def run_inference_thread(file_path):
    infer(file_path)


if __name__ == '__main__':
    folder_data_path = './data'
    csv_output = "./result/triton_python.csv"
    processing_times = []

    file_paths = [os.path.join(folder_data_path, file_path) for file_path in os.listdir(folder_data_path) if file_path.endswith('.wav')]

    for i in range(1, 3):
        start_time = time.time()

        threads = []

        for file_path in file_paths[0:i]:
            thread = threading.Thread(target=run_inference_thread, args=(file_path,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        processing_times.append(end_time - start_time)
        print(i, end_time - start_time)

    df = pd.DataFrame({
        "Time (seconds)": processing_times,
    })
    df.to_csv(csv_output, index=False)

    print("DONE!")
