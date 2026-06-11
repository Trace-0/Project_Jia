from scipy.io.wavfile import write
import re
import time
import uuid
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import onnxruntime
import numpy as np
from typing import Generator, AsyncGenerator, Iterator, cast
from numpy.typing import NDArray
import asyncio
import threading
from typing_extensions import TypedDict, NotRequired, Literal

# Thanks for freddyaboulton/orpheus-cpp repo
# I modified the code to work with Korean TTS and fixed some problems.
CUSTOM_TOKEN_PREFIX = "<custom_token_"

repo_id = "freddyaboulton/3b-ko-ft-research_release-Q4_K_M-GGUF"
file_name = "3b-ko-ft-research_release-q4_k_m.gguf"

model_file = hf_hub_download(
    repo_id=repo_id,
    filename=file_name
)

llm = Llama(
    model_path=model_file,
    n_gpu_layers=30,
    n_ctx=2048,
    verbose=True,
)

class TTSOptions(TypedDict):
    max_tokens: NotRequired[int]
    """Maximum number of tokens to generate. Default: 2048"""
    temperature: NotRequired[float]
    """Temperature for top-p sampling. Default: 0.8"""
    top_p: NotRequired[float]
    """Top-p sampling. Default: 0.95"""
    top_k: NotRequired[int]
    """Top-k sampling. Default: 40"""
    min_p: NotRequired[float]
    """Minimum probability for top-p sampling. Default: 0.05"""
    pre_buffer_size: NotRequired[float]
    """Seconds of audio to generate before yielding the first chunk. Smoother audio streaming at the cost of higher time to wait for the first chunk."""
    voice_id: NotRequired[
        Literal["유나", "준서"]
    ]
    """The voice to use for the TTS. Default: "유나"."""


repo_id = "onnx-community/snac_24khz-ONNX"
snac_model_file = "decoder_model.onnx"
snac_model_path = hf_hub_download(
    repo_id, subfolder="onnx", filename=snac_model_file
)

snac_session = onnxruntime.InferenceSession(
    snac_model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

def token_to_id(token_text: str, index: int) -> int | None:
    token_string = token_text.strip()

    # Find the last token in the string
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

    if last_token_start == -1:
        return None

    # Extract the last token
    last_token = token_string[last_token_start:]

    # Process the last token
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    else:
        return None

def decode(token_gen: Generator[str, None, None]) -> Generator[np.ndarray, None, None]:
    """Asynchronous token decoder that converts token stream to audio stream."""
    buffer = []
    count = 0
    for token_text in token_gen:
        token = token_to_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc)
                if audio_samples is not None:
                    yield audio_samples

def convert_to_audio(multiframe: list[int]) -> np.ndarray | None:
    if len(multiframe) < 28:  # Ensure we have enough tokens
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[: num_frames * 7]

    # Initialize empty numpy arrays instead of torch tensors
    codes_0 = np.array([], dtype=np.int64)
    codes_1 = np.array([], dtype=np.int64)
    codes_2 = np.array([], dtype=np.int64)

    for j in range(num_frames):
        i = 7 * j
        # Append values to numpy arrays
        codes_0 = np.append(codes_0, frame[i])

        codes_1 = np.append(codes_1, [frame[i + 1], frame[i + 4]])

        codes_2 = np.append(
            codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]]
        )

    # Reshape arrays to match the expected input format (add batch dimension)
    codes_0 = np.expand_dims(codes_0, axis=0)
    codes_1 = np.expand_dims(codes_1, axis=0)
    codes_2 = np.expand_dims(codes_2, axis=0)

    # Check that all tokens are between 0 and 4096
    if (
        np.any(codes_0 < 0)
        or np.any(codes_0 > 4096)
        or np.any(codes_1 < 0)
        or np.any(codes_1 > 4096)
        or np.any(codes_2 < 0)
        or np.any(codes_2 > 4096)
    ):
        return None

    # Create input dictionary for ONNX session

    snac_input_names = [x.name for x in snac_session.get_inputs()]

    input_dict = dict(zip(snac_input_names, [codes_0, codes_1, codes_2]))

    # Run inference
    audio_hat = snac_session.run(None, input_dict)[0]

    # Process output
    audio_np = audio_hat[:, :, 2048:4096]
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    return audio_bytes

def tts(text: str, options: TTSOptions | None = None) -> tuple[int, NDArray[np.int16]]:
    buffer = []
    for _, array in stream_tts_sync(text, options):
        buffer.append(array)
    return (24_000, np.concatenate(buffer, axis=1))

async def stream_tts(text: str, options: TTSOptions | None = None) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
    queue = asyncio.Queue()
    finished = asyncio.Event()

    def strem_to_queue(text, options, queue, finished):
        for chunk in stream_tts_sync(text, options):
            queue.put_nowait(chunk)
        finished.set()

    thread = threading.Thread(
        target=strem_to_queue, args=(text, options, queue, finished)
    )
    thread.start()
    while not finished.is_set():
        try:
            yield await asyncio.wait_for(queue.get(), 0.1)
        except (asyncio.TimeoutError, TimeoutError):
            pass
    while not queue.empty():
        chunk = queue.get_nowait()
        yield chunk

def token_gen(text: str, options: TTSOptions | None = None) -> Generator[str, None, None]:
    from llama_cpp import CreateCompletionStreamResponse

    options = options or TTSOptions()
    voice_id = options.get("voice_id", "유나")
    text = f"<|audio|><custom_token_128259>{voice_id}: {text}<|eot_id|>"
    token_gen = llm(
        text,
        max_tokens=options.get("max_tokens", 2048),
        stream=True,
        temperature=options.get("temperature", 0.6),
        repeat_penalty=1.3,
        stop=["<custom_token_49158>"]
    )
    for token in cast(Iterator[CreateCompletionStreamResponse], token_gen):
        yield token["choices"][0]["text"]

def stream_tts_sync(text: str, options: TTSOptions | None = None) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
    options = options or TTSOptions()
    token_gene = token_gen(text, options)
    pre_buffer = np.array([], dtype=np.int16).reshape(1, 0)
    pre_buffer_size = 24_000 * options.get("pre_buffer_size", 1.5)
    started_playback = False
    for audio_bytes in decode(token_gene):
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
        if not started_playback:
            pre_buffer = np.concatenate([pre_buffer, audio_array], axis=1)
            if pre_buffer.shape[1] >= pre_buffer_size:
                started_playback = True
                yield (24_000, pre_buffer)
        else:
            yield (24_000, audio_array)
    if not started_playback:
        yield (24_000, pre_buffer)

def generate_with_orpheus_tts(text):
    text = re.sub(r'[^A-Za-z가-힣\s.?!]', '', text)
    print(text)
    samlple_rate, samples = tts(text, options={"pre_buffer_size": 0, "max_tokens": 2048, "temperature": 0.6, "voice_id": "유나"})

    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex
    file_name = f"output_temp/output_{timestamp}_{unique_id}.wav"

    write(file_name, samlple_rate, samples.squeeze())

if __name__ == "__main__":
    test_text = "안녕. 만나서 반가워. 너 이름이 뭐니?"
    generate_with_orpheus_tts(test_text)

# 한숨, 헐, 헛기침, 훌쩍, 하품, 낄낄, 신음, 작은 웃음, 기침, 으르렁
# 위의 단어가 포함되면 모델에 문제가 발생해서 환각 현상이 발생함
# 위 단어가 포함되면 안됨. 부분적으로 응답을 제거하거나 시스템 프롬프트로 위 단어를 사용하지 말라고 지시해야 함.
# 딜레이가 너무 길어서 생성해놓고 재생하는 방식은 너무 느림. 실시간 스트리밍이 필요함.
# 디스코드 봇과 langchain에서 실시간 스트리밍이 가능한지 확인 필요.
# 환각 현상이 너무 심함. 다른 모델을 찾아보는게 좋을 듯.