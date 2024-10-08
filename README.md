# run

1. ```sudo docker build -t my_triton_server .```
2. ```sudo docker run --name my_triton_server --gpus all --shm-size=1g -p 8000:8000 -p 8001:8001 -p 8002:8002 -ti my_triton_server```
3. ```sudo docker exec -it my_triton_server```
4. ```sudo docker exec -it fe6131026c1a perf_analyzer -m wav2vec_process --percentile=95 --concurrency-range 1:16 --input-data zero --shape sample_rate:1 --shape waveform:1,32000 --measurement-interval=20000```