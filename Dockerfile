
FROM nvcr.io/nvidia/tritonserver:23.01-py3

RUN pip install torch 
RUN pip install transformers 
RUN pip install numpy 
RUN pip install pydub 
RUN pip install torchaudio 
RUN pip install onnx
RUN python3 -m pip install --upgrade pip
RUN pip install tritonclient[http]
RUN pip install requests
RUN git clone https://github.com/triton-inference-server/python_backend -b r23.01

RUN cd python_backend

COPY ./models /models


ENV PYTHONPATH="/opt/tritonserver/backends/python:$PYTHONPATH"


CMD ["tritonserver", "--model-repository=/models"]