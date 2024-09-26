import time
from concurrent.futures import ThreadPoolExecutor
import torch
import torchaudio
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def preprocess_audio(file_path: str):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio = AudioSegment.from_wav(file_path)

    samples = audio.get_array_of_samples()
    waveform = torch.tensor(samples).float().unsqueeze(0)  
    sample_rate = audio.frame_rate

    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)

    waveform = waveform[:, :16000]

    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

    return input_values


file_paths = [
    '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt0.wav',
    '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt1.wav',
    '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt2.wav',
    '/home/trandat/Documents/vnpt/test-docker/data/dia0_utt3.wav'
]


def process_files_concurrently(file_paths):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(preprocess_audio, file_path) for file_path in file_paths]
        input_values_list = [future.result() for future in futures]
    
    # Gộp các input_values thành một batch
    batched_input_values = torch.cat(input_values_list, dim=0)

    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    with torch.no_grad():
        output = model(batched_input_values).last_hidden_state

    print(output, output.shape)

    end_time = time.time()
    print(f"Total time for concurrent execution: {end_time - start_time:.2f} seconds")
    

    return output


results = process_files_concurrently(file_paths)
