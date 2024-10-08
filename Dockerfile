
FROM nvcr.io/nvidia/tritonserver:23.09-py3

RUN pip install torch 
RUN pip install transformers 
RUN pip install numpy 
RUN pip install pydub 
RUN pip install torchaudio 
RUN pip install onnx
RUN pip install tritonclient[all]
RUN pip install requests
RUN git clone https://github.com/triton-inference-server/python_backend -b r23.01

RUN cd python_backend

COPY ./models /models

CMD ["tritonserver", "--model-repository=/models"]