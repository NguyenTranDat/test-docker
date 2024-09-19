import os
import torch
import torchaudio
from pydub import AudioSegment
import requests
from transformers import Wav2Vec2Processor
from flask import Flask, request, jsonify

app = Flask(__name__)

TRITON_SERVER_URL = "http://triton_server:8000/v2/models/wav2vec2_model/infer"


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


def send_to_triton(input_values):
    input_data = input_values.numpy().astype('float32').tolist()
    data = {
        "inputs": [
            {
                "name": "input",
                "shape": input_values.shape,
                "datatype": "FP32",
                "data": input_data
            }
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(TRITON_SERVER_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


@app.route('/extract_features', methods=['POST'])
def extract_features():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join("/tmp", file.filename)
    file.save(file_path)

    if not os.path.isfile(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 500

    try:
        input_values = preprocess_audio(file_path)
        triton_response = send_to_triton(input_values)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(triton_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
