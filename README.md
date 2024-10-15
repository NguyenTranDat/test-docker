# run

1. ```sudo docker build -t test .```
2. ```sudo docker run --name test --gpus all --shm-size=1g -p 8000:8000 -p 8001:8001 -p 8002:8002 test```
4. ```sudo docker exec test perf_analyzer -m wav2vec_py --concurrency-range 1:16 --input-data zero --shape sample_rate:1 --shape waveform:32000 --measurement-interval=20000```