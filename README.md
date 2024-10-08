# run

1. ```sudo docker build -t my_triton_server .```
2. ```sudo docker run --name my_triton_server --gpus all --shm-size=1g -p 8000:8000 -p 8001:8001 -p 8002:8002 -ti my_triton_server```
3. ```sudo docker exec -it my_triton_server```
4. ```perf_analyzer -m wav2vec_py --percentile=95 --concurrency-range 1:8 --input-data /util/test.json --input-tensor-format json```