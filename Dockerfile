FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN python3 -m pip install --upgrade pip
RUN pip install torch 
RUN pip install transformers 
RUN pip install numpy 
RUN pip install pydub 
RUN pip install torchaudio 
RUN pip install onnx
RUN pip install tritonclient[all]
RUN pip install requests
RUN git clone https://github.com/triton-inference-server/python_backend -b r23.10

RUN cd python_backend

COPY ./models /models

CMD ["tritonserver", "--model-repository=/models", "--log-format=default", "--log-file=/opt/tritonserver/logfile.log"] 
# CMD ["tritonserver", "--model-repository=/models", "--cache-config=local,size=1000048576", "--log-verbose=1"]   
# CMD ["tritonserver", "--model-repository=/models", "--log-format=default", "--log-file=/opt/tritonserver/logfile.log"]   
# CMD ["tritonserver", "--model-repository=/models", "--log-verbose=2"]   