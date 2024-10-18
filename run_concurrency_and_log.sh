#!/bin/bash

for concurrency in {1..16}
do
   echo "Running perf_analyzer with concurrency=$concurrency"

   sudo docker exec e42c3e2a5a75 perf_analyzer -m wav2vec_py --concurrency-range $concurrency:$concurrency --input-data zero --shape sample_rate:1 --shape waveform:32000 --measurement-interval=20000
   
   echo "Logging stats after concurrency=$concurrency"
   curl -X GET "http://localhost:8000/v2/models/wav2vec_py/versions/1/stats"
done
