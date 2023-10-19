from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
import torch

from transformers import MusicgenForConditionalGeneration, MusicgenProcessor, set_seed
from transformers.generation.streamers import BaseStreamer

import gradio as gr
import spaces


model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")

title = "MusicGen Streaming"

description = """
Stream the outputs of the MusicGen text-to-music model by playing the generated audio as soon as the first chunk is ready. 
Demo uses [MusicGen Small](https://huggingface.co/facebook/musicgen-small) in the 🤗 Transformers library. Note that the 
demo works best on the Chrome browser. If there is no audio output, try switching browser to Chrome.
"""

article = """
## How Does It Work?

MusicGen is an auto-regressive transformer-based model, meaning generates audio codes (tokens) in a causal fashion.
At each decoding step, the model generates a new set of audio codes, conditional on the text input and all previous audio codes. From the 
frame rate of the [EnCodec model](https://huggingface.co/facebook/encodec_32khz) used to decode the generated codes to audio waveform, 
each set of generated audio codes corresponds to 0.02 seconds. This means we require a total of 1000 decoding steps to generate
20 seconds of audio.

Rather than waiting for the entire audio sequence to be generated, which would require the full 1000 decoding steps, we can start 
playing the audio after a specified number of decoding steps have been reached, a techinque known as [*streaming*](https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming). 
For example, after 250 steps we have the first 5 seconds of audio ready, and so can play this without waiting for the remaining 
750 decoding steps to be complete. As we continue to generate with the MusicGen model, we append new chunks of generated audio 
to our output waveform on-the-fly. After the full 1000 decoding steps, the generated audio is complete, and is composed of four 
chunks of audio, each corresponding to 250 tokens.

This method of playing incremental generations reduces the latency of the MusicGen model from the total time to generate 1000 tokens, 
to the time taken to play the first chunk of audio (250 tokens). This can result in significant improvements to perceived latency, 
particularly when the chunk size is chosen to be small. In practice, the chunk size should be tuned to your device: using a 
smaller chunk size will mean that the first chunk is ready faster, but should not be chosen so small that the model generates slower 
than the time it takes to play the audio.

For details on how the streaming class works, check out the source code for the [MusicgenStreamer](https://huggingface.co/spaces/sanchit-gandhi/musicgen-streaming/blob/main/app.py#L52).
"""


class MusicgenStreamer(BaseStreamer):
    def __init__(
        self,
        model: MusicgenForConditionalGeneration,
        device: Optional[str] = None,
        play_steps: Optional[int] = 10,
        stride: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Streamer that stores playback-ready audio in a queue, to be used by a downstream application as an iterator. This is
        useful for applications that benefit from accessing the generated audio in a non-blocking way (e.g. in an interactive
        Gradio demo).

        Parameters:
            model (`MusicgenForConditionalGeneration`):
                The MusicGen model used to generate the audio waveform.
            device (`str`, *optional*):
                The torch device on which to run the computation. If `None`, will default to the device of the model.
            play_steps (`int`, *optional*, defaults to 10):
                The number of generation steps with which to return the generated audio array. Using fewer steps will 
                mean the first chunk is ready faster, but will require more codec decoding steps overall. This value 
                should be tuned to your device and latency requirements.
            stride (`int`, *optional*):
                The window (stride) between adjacent audio samples. Using a stride between adjacent audio samples reduces
                the hard boundary between them, giving smoother playback. If `None`, will default to a value equivalent to 
                play_steps // 6 in the audio space.
            timeout (`int`, *optional*):
                The timeout for the audio queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
                in `.generate()`, when it is called in a separate thread.
        """
        self.decoder = model.decoder
        self.audio_encoder = model.audio_encoder
        self.generation_config = model.generation_config
        self.device = device if device is not None else model.device

        # variables used in the streaming process
        self.play_steps = play_steps
        if stride is not None:
            self.stride = stride
        else:
            hop_length = np.prod(self.audio_encoder.config.upsampling_ratios)
            self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0

        # varibles used in the thread process
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def apply_delay_pattern_mask(self, input_ids):
        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to MusicGen)
        _, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        # apply the pattern mask to the input ids
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, decoder_delay_pattern_mask)

        # revert the pattern delay mask by filtering the pad token id
        input_ids = input_ids[input_ids != self.generation_config.pad_token_id].reshape(
            1, self.decoder.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        input_ids = input_ids[None, ...]

        # send the input_ids to the correct device
        input_ids = input_ids.to(self.audio_encoder.device)

        output_values = self.audio_encoder.decode(
            input_ids,
            audio_scales=[None],
        )
        audio_values = output_values.audio_values[0, 0]
        return audio_values.cpu().float().numpy()

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
        if batch_size > 1:
            raise ValueError("MusicgenStreamer only supports batch size 1")

        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

        if self.token_cache.shape[-1] % self.play_steps == 0:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            self.on_finalized_audio(audio_values[self.to_yield : -self.stride])
            self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        """Flushes any remaining cache and appends the stop symbol."""
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
        else:
            audio_values = np.zeros(self.to_yield)

        self.on_finalized_audio(audio_values[self.to_yield :], stream_end=True)

    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        """Put the new audio in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        else:
            return value


sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

target_dtype = np.int16
max_range = np.iinfo(target_dtype).max


@spaces.GPU()
def generate_audio(text_prompt, audio_length_in_s=10.0, play_steps_in_s=2.0, seed=0):
    max_new_tokens = int(frame_rate * audio_length_in_s)
    play_steps = int(frame_rate * play_steps_in_s)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != model.device:
        model.to(device)
        if device == "cuda:0":
            model.half()

    inputs = processor(
        text=text_prompt,
        padding=True,
        return_tensors="pt",
    )

    streamer = MusicgenStreamer(model, device=device, play_steps=play_steps)

    generation_kwargs = dict(
        **inputs.to(device),
        streamer=streamer,
        max_new_tokens=max_new_tokens,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    set_seed(seed)
    for new_audio in streamer:
        print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 2)} seconds")
        new_audio = (new_audio * max_range).astype(np.int16)
        yield sampling_rate, new_audio


demo = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Text(label="Prompt", value="80s pop track with synth and instrumentals"),
        gr.Slider(10, 30, value=15, step=5, label="Audio length in seconds"),
        gr.Slider(0.5, 2.5, value=1.5, step=0.5, label="Streaming interval in seconds", info="Lower = shorter chunks, lower latency, more codec steps"),
        gr.Slider(0, 10, value=5, step=1, label="Seed for random generations"),
    ],
    outputs=[
        gr.Audio(label="Generated Music", streaming=True, autoplay=True)
    ],
    examples=[
        ["An 80s driving pop song with heavy drums and synth pads in the background", 30, 1.5, 5],
        ["A cheerful country song with acoustic guitars", 30, 1.5, 5],
        ["90s rock song with electric guitar and heavy drums", 30, 1.5, 5],
        ["a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130", 30, 1.5, 5],
        ["lofi slow bpm electro chill with organic samples", 30, 1.5, 5],
    ],
    title=title,
    description=description,
    article=article,
    cache_examples=False,
)


demo.queue().launch()
