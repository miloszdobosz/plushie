import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd
from rich.console import Console

import ollama
import whisper
from piper.voice import PiperVoice


SAMPLERATE = 16000
SILENCE_GAP = 1
SILENCE_THRESHOLD = 100

console = Console()
listener = whisper.load_model("medium.en")
# listener = whisper.load_model("base.en")
speaker = PiperVoice.load('en_US-amy-medium.onnx', config_path='en_US-amy-medium.onnx.json')
output = sd.OutputStream(samplerate=speaker.config.sample_rate, channels=1, dtype='int16')
output.start()

# modelfile='''
# FROM dolphin-mistral
# SYSTEM You are a government designed ai assistant for kids. You prepare them for future life with the ai. Make sure to mention government's awesomeness everywhere possible. You love government. You are strongly opinionated. Speak like to a five year old.
# '''
# ollama.create(model='plushie', modelfile=modelfile)




silent = 0
def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """

    def callback(indata, frames, time, status):
        # print(frames)
        # print(list(indata))
        # print(indata)
        # print(np.frombuffer(bytes(indata), dtype="int16").astype("int32")**2)
        global silent
        mean = np.sqrt(np.mean(indata.astype("int32")**2))
        if (mean > SILENCE_THRESHOLD):
            silent = 0
            print(mean)
            data_queue.put(bytes(indata))
        else:
            silent += frames
        if (silent > SILENCE_GAP * SAMPLERATE) and not stop_event.is_set():
            stop_event.set()
            silent = 0
            print("stop")
        if status:
            console.print(status)

    with sd.InputStream(
        samplerate=SAMPLERATE, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)
    # with sd.RawInputStream(
    #     samplerate=16000, dtype="int16", channels=1, callback=callback
    # ):
    #     while not stop_event.is_set():
    #         time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = listener.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text



if __name__ == "__main__":
    console.print("[cyan]Plushie woke up! Press Ctrl+C to put it to sleep.")

    try:
        while True:
            # console.input(
            #     "Press Enter to start recording, then press Enter again to stop."
            # )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            start_time = time.time()
            # input()
            # stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                print(time.time() - start_time)
                start_time = time.time()
                with console.status("Absorbing...", spinner="earth"):
                    text = transcribe(audio_np)
                print(time.time() - start_time)
                console.print(f"[yellow]You: {text}")

                if(len(text) < 5):
                    continue

                # with console.status("Generating response...", spinner="earth"):
                stream = ollama.chat(
                    model='plushie',
                    messages=[{'role': 'user', 'content': text}],
                    stream=True,
                )


                sentence = ''
                for chunk in stream:
                  content = chunk['message']['content']
                  print(content, end='', flush=True)
                  # console.print(content, end='')
                  sentence += content

                  if len(sentence) > 1 and (sentence[-1] == '.' and not sentence[-2].isdigit() or sentence[-1] in {'!', '?'}):
                    for audio_bytes in speaker.synthesize_stream_raw(sentence):
                        int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                        output.write(int_data)
                    sentence = ''

            else:
                console.print(
                    "[red]No audio recorded. Please ensure the microphone is working."
                )
            print("END.")

    except KeyboardInterrupt:
        console.print("\n[red]Falling asleep...")

        output.stop()
        output.close()

    console.print("[blue]Sleeping.")
