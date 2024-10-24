import threading
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import time
import os
from pydub import AudioSegment
import torch
import pandas as pd
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

MODEL_NAME = 'wav2vec_py'
TRITON_SERVER_URL = f'http://localhost:8000/v2/models/{MODEL_NAME}/infer'

@pipeline_def
def audio_decoder_pipe(file_paths):
    encoded, _ = fn.readers.file(files=file_paths, file_filters="*.wav")
    audio, sampling_rate = fn.decoders.audio(encoded, dtype=types.INT16)
    audio = fn.audio_resample(audio, in_rate=sampling_rate, out_rate=16000)
    return audio

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

if __name__ == '__main__':
    folder_data_path = '/home/trandat/Documents/vnpt/test-docker/data'
    csv_output = "./results/triton_python.csv"
    max_workers = 4
    processing_times = []
    file_counts = []

    file_paths = [os.path.join(folder_data_path, f) for f in os.listdir(folder_data_path) if f.endswith('.wav')]

    for i in range(1, 17):
        start_time = time.time()

        threads = []

        pipe = audio_decoder_pipe(batch_size=i, num_threads=16, device_id=0, file_paths=file_paths[:i])
        pipe.build()
        waveforms = pipe.run()

        for j in range(i):
            waveform = waveforms[0].at(j).flatten()
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
        "File Count": file_counts,
        "Time (seconds)": processing_times,
    })
    df.to_csv(csv_output, index=False)

    print("DONE!")
