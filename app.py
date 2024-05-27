import os
import shutil
import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd
import wave
from rich.console import Console

import ollama
import whisper
from piper.voice import PiperVoice


SAMPLERATE = 16000
MIN_DURATION = 1
SILENCE_GAP = 1
SILENCE_THRESHOLD = 100

console = Console()
listener = whisper.load_model("medium.en")
speaker = PiperVoice.load('en_US-amy-medium.onnx', config_path='en_US-amy-medium.onnx.json')
output = sd.OutputStream(samplerate=speaker.config.sample_rate, channels=1, dtype='int16')
output.start()


def record_audio(stop_event, recorded_queue):
    stream = Queue()
    silent_samples = 0
    loud_samples = 0

    def callback(indata, frames, _time, _status):
        nonlocal stream, silent_samples, loud_samples

        # RMS volume
        volume = np.sqrt(np.mean(np.frombuffer(indata, dtype="int16").astype("int32")**2))

        if (volume > SILENCE_THRESHOLD):
            console.print(f'[green]\r{"█" * int(volume / 200)}')

            silent_samples = 0
            loud_samples += frames

            stream.put(bytes(indata)) 

        else:
            silent_samples += frames

            if (silent_samples > SILENCE_GAP * SAMPLERATE):
                # Cut a prompt
                audio_data = b"".join(stream.queue)
                
                if loud_samples > MIN_DURATION * SAMPLERATE:
                    recorded_queue.put(np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0)
                    loud_samples = 0
                
                silent_samples = 0
                stream = Queue()

    with sd.RawInputStream(samplerate=SAMPLERATE, dtype="int16", channels=1, callback=callback):
        stop_event.wait()


def transcribe(audio_np: np.ndarray) -> str:
    result = listener.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    return result["text"].strip()


def say(text: str):
    for audio_bytes in speaker.synthesize_stream_raw(text):
        output.write(np.frombuffer(audio_bytes, dtype=np.int16))


def is_sentence(text: str) -> bool:
    return len(text) > 1 and (text[-1] == '.' and not text[-2].isdigit() or text[-1] in {'!', '?'})


if __name__ == "__main__":
    console.print("[cyan]Plushie woke up! Press Ctrl+C to put it back to sleep.")

    stop_event = None
    recording_thread = None

    try:
        while True:
            stop_event = threading.Event()
            recorded_queue = Queue()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event, recorded_queue))
            recording_thread.start()

            text = ""
            while True:
                audio = recorded_queue.get()
                with console.status("[green]Thinking", spinner="point"):
                    text = transcribe(audio)

                    console.print(f"[green]\nYou:\n [yellow]{text}")

                    if(len(text) > 4):
                        break

                console.print("[red]Ignoring short prompt (probably noise)")


            stop_event.set()

            stream = ollama.chat(
                model='plushie',
                messages=[{'role': 'user', 'content': text}],
                stream=True,
            )
            sentence = ''

            console.print("[green]Plushie:", end="\n ")
            for chunk in stream:
                content = chunk['message']['content']
                console.print(f"[cyan]{content}", end='')

                sentence += content

                if is_sentence(sentence):
                    console.print()

                    with console.status("[green]Thinking", spinner="point"):
                        say(sentence)
                        sentence = ''

    except KeyboardInterrupt:
        with console.status("\n[green]Falling asleep", spinner="point"):
            output.stop()
            output.close()

            stop_event.set()
            recording_thread.join()

    console.print("[green]\nSleeping.")
