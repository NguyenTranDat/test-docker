
import numpy as np
import torch
from pydub import AudioSegment
from tritonclient.utils import *
import tritonclient.http as httpclient
import concurrent.futures
import time
import os
import pandas as pd

model_name = "wav2vec_process"

def run_inference(file_path):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        audio = AudioSegment.from_wav(file_path)
        samples = audio.get_array_of_samples()
        waveform = torch.tensor(samples).float().unsqueeze(0)  
        sample_rate = audio.frame_rate

        waveform = waveform.numpy()
        sample_rate = np.array([sample_rate], dtype=np.float32)

        inputs = [
            httpclient.InferInput("waveform", waveform.shape, np_to_triton_dtype(waveform.dtype)),
            httpclient.InferInput("sample_rate", sample_rate.shape, np_to_triton_dtype(sample_rate.dtype))
        ]

        inputs[0].set_data_from_numpy(waveform)
        inputs[1].set_data_from_numpy(sample_rate)

        outputs = [
            httpclient.InferRequestedOutput("output"),
        ]

        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        result = response.get_response()
        output_data = response.as_numpy("output")

        # print(f"Response for {file_path}:", result)
        # print(f"Output Data for {file_path}:", output_data)

if __name__ == '__main__':
    csv_output="./results/triton_client.csv"
    folder_data_path = './data'
    max_workers = 4
    processing_times = []
    file_counts = []

    file_paths = [os.path.join(folder_data_path, file_path) for file_path in os.listdir(folder_data_path) if file_path.endswith('.wav')]

    for i in range(1,33):
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_inference, file_paths[0:i])

        end_time = time.time()

        processing_times.append(end_time-start_time)
        file_counts.append(i) 

    df = pd.DataFrame({
        "File Count": file_counts,
        "Time (seconds)": processing_times,
    })
    df.to_csv(csv_output, index=False)

    print("DONE!")

    print(f"Total time taken: {end_time - start_time:.2f} seconds")
