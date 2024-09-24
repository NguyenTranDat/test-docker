import numpy as np
import torch
from pydub import AudioSegment
from tritonclient.utils import *
import tritonclient.http as httpclient
import threading
import time

model_name = "wav2vec_process"

def preprocess_audio(file_path: str):
    audio = AudioSegment.from_wav(file_path)
    samples = audio.get_array_of_samples()
    waveform = torch.tensor(samples).float().unsqueeze(0)  
    sample_rate = audio.frame_rate

    return waveform, sample_rate

def run_inference(file_path):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        waveform, sample_rate = preprocess_audio(file_path)
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

        response = client.infer(model_name,
                                 inputs,
                                 request_id=str(1),
                                 outputs=outputs)

        result = response.get_response()
        output_data = response.as_numpy("output")

        print(f"Response for {file_path}:", result)
        print(f"Output Data for {file_path}:", output_data)

def run():
    file_paths = [
        '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt0.wav',
        '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt1.wav',
        '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt2.wav',
        '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt3.wav'
    ]

    threads = []
    start_time = time.time()

    for file_path in file_paths:
        thread = threading.Thread(target=run_inference, args=(file_path,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

run()
