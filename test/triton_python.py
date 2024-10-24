import threading
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import time
import os
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
            httpclient.InferInput("waveform", waveform.shape, np_to_triton_dtype(waveform.dtype)),
        ]

        inputs[0].set_data_from_numpy(waveform)

        outputs = [
            httpclient.InferRequestedOutput("output"),
        ]

        response = client.infer(MODEL_NAME, inputs, request_id=str(1), outputs=outputs)

        output_data = response.as_numpy("output")

if __name__ == '__main__':
    folder_data_path = '/home/admin1/Documents/test-docker/data'
    csv_output = "./results/triton_python.csv"
    max_workers = 4
    processing_times = []
    file_counts = []

    file_paths = [os.path.join(folder_data_path, f) for f in os.listdir(folder_data_path) if f.endswith('.wav')]


    for i in range(2, 3):
        start_time = time.time()

        threads = []

        pipe = audio_decoder_pipe(batch_size=i, num_threads=16, device_id=0, file_paths=file_paths[:i])
        pipe.build()
        waveform = pipe.run()

        print(waveform)

        threads = []
        waveforms = []
        for w in waveform:
            w = w.at(0)
            waveforms.append(w)

        print(waveforms)
        
        thread = threading.Thread(target=infer, args=(waveforms,))
        threads.append(thread)
        thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        processing_times.append(end_time - start_time)
        file_counts.append(i)

    df = pd.DataFrame({
        "File Count": file_counts,
        "Time (seconds)": processing_times,
    })
    df.to_csv(csv_output, index=False)

    print("DONE!")
