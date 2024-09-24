# run

1. ```sudo docker build -t my_triton_server .```
2. ```sudo docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti my_triton_server```

# Test
1. ```python test/trition.py```
