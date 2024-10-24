import threading
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import time
import os
from pydub import AudioSegment
import torch
import pandas as pd

MODEL_NAME = 'wav2vec_py'
TRITON_SERVER_URL = f'http://localhost:8000/v2/models/{MODEL_NAME}/infer'

def read_audio_file(file_path):
    audio = AudioSegment.from_wav(file_path)
    audio = audio.set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return samples

def infer(waveform):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        inputs = [
            httpclient.InferInput("waveform", waveform.shape, np_to_triton_dtype(np.float32)),
        ]

        inputs[0].set_data_from_numpy(waveform.astype(np.float32))

        outputs = [
            httpclient.InferRequestedOutput("output"),
        ]

        response = client.infer(MODEL_NAME, inputs, request_id='1', outputs=outputs)
        output_data = response.as_numpy("output")
        return output_data

if __name__ == '__main__':
    folder_data_path = '/home/trandat/Documents/vnpt/test-docker/data'
    csv_output = "./results/triton_python_2.csv"
    processing_times = []
    file_counts = []

    file_paths = [os.path.join(folder_data_path, f) for f in os.listdir(folder_data_path) if f.endswith('.wav')]

    for i in range(1, 17):
        start_time = time.time()

        threads = []
        waveforms = []

        for j in range(i):
            waveform = read_audio_file(file_paths[j])
            waveform = waveform.flatten()
            thread = threading.Thread(target=infer, args=(np.array([waveform]),))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        processing_times.append(end_time - start_time)
        print(i, end_time - start_time)
        file_counts.append(i)

    df = pd.DataFrame({
        "Time (seconds)": processing_times,
    })
    df.to_csv(csv_output, index=False)

    print("DONE!")
