import time
import os
import concurrent.futures
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
from pydub import AudioSegment
import torch.nn.functional as F
import pandas as pd


def preprocess(file_path: str):
    audio = AudioSegment.from_wav(file_path)
    samples = audio.get_array_of_samples()
    waveform = torch.tensor(samples).float().unsqueeze(0)  
    sample_rate = audio.frame_rate

    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    features = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

    return features


def pad_tensors(tensor_list):
    max_length = max(tensor.shape[1] for tensor in tensor_list)
    padded_tensors = [F.pad(tensor, (0, max_length - tensor.shape[1])) for tensor in tensor_list]
    return torch.cat(padded_tensors, dim=0)


def process_files_concurrently():
    folder_data_path = './data'
    max_threads = 4
    processing_times = []
    file_counts = []

    file_paths = [os.path.join(folder_data_path, file) for file in os.listdir(folder_data_path) if file.endswith('.wav')]

    for i in range(1,33):
        start_time = time.time()
        split_file_paths = file_paths[1:i]

        batches = [split_file_paths[i:i + max_threads] for i in range(0, len(split_file_paths), max_threads)]
        
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        for batch in batches:
            input_values_list = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                input_values_list = list(executor.map(preprocess, batch))

            batched_input_values = pad_tensors(input_values_list)

            with torch.no_grad():
                output = model(batched_input_values).last_hidden_state

            # print(f"Processed batch of size {len(batch)}:")
            # print(output, output.shape)

        end_time = time.time()
        processing_times.append(end_time-start_time)
        file_counts.append(i) 

    csv_output="./result/model2.csv"

    df = pd.DataFrame({
        "File Count": file_counts,
        "Time (seconds)": processing_times,
    })
    df.to_csv(csv_output, index=False)

    print("DONE!")


if __name__ == '__main__':
    process_files_concurrently()
