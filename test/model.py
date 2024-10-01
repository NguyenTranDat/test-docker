import time
import threading
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
from pydub import AudioSegment


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

    print(output, output.shape)

    return output


def process_files_concurrently():
    file_paths = [
        './data/dia0_utt0.wav',
        './data/dia0_utt1.wav',
        './data/dia0_utt2.wav',
        './data/dia0_utt3.wav'
    ]
    start_time = time.time()

    threads = []
    for file_path in file_paths:
        thread = threading.Thread(target=preprocess_audio, args=(file_path,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    
    end_time = time.time()
    print(f"Total time for concurrent execution: {end_time - start_time:.2f} seconds")

process_files_concurrently()
