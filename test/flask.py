import requests
import time
from concurrent.futures import ThreadPoolExecutor

def send_file_to_flask(file_path):
    url = "http://localhost:5000/extract_features"
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url, files=files)
    return response.json()

file_paths = [
    '/home/trandat/test_docker/data/dia0_utt0.wav',
    '/home/trandat/test_docker/data/dia0_utt1.wav',
    '/home/trandat/test_docker/data/dia0_utt2.wav',
    '/home/trandat/test_docker/data/dia0_utt3.wav'
]

def send_files_concurrently(file_paths):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_file_to_flask, file_path) for file_path in file_paths]
        for future in futures:
            try:
                result = future.result()
                print(f"Response: {result}")
            except Exception as e:
                print(f"Error occurred: {str(e)}")

    end_time = time.time()
    print(f"Total time for concurrent execution: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    send_files_concurrently(file_paths)
