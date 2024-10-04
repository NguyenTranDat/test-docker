import time
import os
import concurrent.futures
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
from pydub import AudioSegment
import pandas as pd


def preprocess_audio(file_path: str):
    audio = AudioSegment.from_wav(file_path)
    samples = audio.get_array_of_samples()
    waveform = torch.tensor(samples).float().unsqueeze(0)  
    sample_rate = audio.frame_rate

    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    features = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    with torch.no_grad():
        output = model(features).last_hidden_state

    del audio, samples, waveform, features, output
    torch.cuda.empty_cache()

    # print(output, output.shape)

    # return output


if __name__ == '__main__':
    folder_data_path = './data'
    csv_output="./result/model.csv"
    max_workers = 4
    processing_times = []

    file_paths = [os.path.join(folder_data_path, file_path) for file_path in os.listdir(folder_data_path) if file_path.endswith('.wav')]

    for i in range(1, 33):
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(preprocess_audio, file_paths[0:i])

        end_time = time.time()

        processing_times.append(end_time-start_time)
        print(i, end_time-start_time)

    df = pd.DataFrame({
        "Time (seconds)": processing_times,
    })
    df.to_csv(csv_output, index=False)

    print("DONE!")

    # print(f"Total time taken: {end_time - start_time:.2f} seconds")
