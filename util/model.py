from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def audio_decoder_pipe(file_path: str):
    encoded, _ = fn.readers.file(file=file_path)
    audio, sampling_rate = fn.decoders.audio(encoded, dtype=types.INT16)
    audio = fn.resample(audio, in_rate=sampling_rate, out_rate=16000)

    return audio