import tensorrt as trt
import os
import onnx

output_dir = "./models/wav2vec_model/1"
onnx_model_path = os.path.join(output_dir, "model.onnx")
onnx_model = onnx.load(onnx_model_path)
engine_file = os.path.join(output_dir, "model.trt")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)


with open(onnx_model_path, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))


profile = builder.create_optimization_profile()
profile.set_shape("input_name", (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))


config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.add_optimization_profile(profile)
serialized_engine = builder.build_serialized_network(network, config)

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)


with open(engine_file, 'wb') as f:
    f.write(engine.serialize())

