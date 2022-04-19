import argparse
import glob
import json
import os
import re
import time
import wave
from io import BytesIO
from time import sleep

import ffmpeg
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from denoiser import enhance


def _get_audio_bytes(audio_filename):
    # audio_bytes = BytesIO(normalize_audio(audio_filename))
    y, sr = librosa.load(audio_filename, sr=16000, mono=True, dtype=np.float32)

    return y
    #
    # with wave.Wave_read(audio_bytes) as wav:
    #     x = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
    #     return x


def np_chunks(lst: np.array, chunk_len: int):
    for i in range(0, len(lst), chunk_len):
        yield lst[i:i + chunk_len]


def normalize_audio(input_file):
    try:
        kwargs = {
            "f": "WAV",
            "acodec": "pcm_s16le",
            "ac": 1,
            "ar": "16k",
            "loglevel": "error",
            "hide_banner": None,
        }

        out, err = ffmpeg.input(input_file).output(
            "pipe:1",
            **kwargs
        ).run(capture_stdout=True, capture_stderr=True)

        return out
    except Exception as e:
        print(e)
        print(e.stderr)


class SpeechToTextEngine:
    def __init__(self, model_name):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        if "hubert" in model_name:
            self.model = HubertForCTC.from_pretrained(model_name)
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.model.to(torch.device("cuda:0"))

    def run(self, audio_filename):
        audio_bytes = _get_audio_bytes(audio_filename)

        text = ""

        # Tweak this number to avoid OOM
        for chunk in np_chunks(audio_bytes, 100_000):
            input_values = self.processor(
                chunk, return_tensors="pt", sampling_rate=16000
            ).input_values

            input_values = input_values.float().to(torch.device("cuda:0"))

            predicted_ids = torch.argmax(self.model(input_values).logits, dim=-1)

            text += self.processor.batch_decode(predicted_ids)[0]
            text += "\n"

        return re.sub(r"\s+", " ", text).lower()


class DenoiseArgs:
    def __init__(self):
        self.device = "cuda"
        self.dry = 0
        self.sample_rate = 16000
        self.num_workers = 32
        self.streaming = True
        self.noisy_dir = "/tmp/transcribe-denoise-input/"
        self.noisy_json = None
        self.dns64 = False
        self.dns48 = False
        self.master64 = True
        self.model_path = None


DENOISE_INPUT = "/tmp/transcribe-denoise-input/"
DENOISE_OUTPUT = "/tmp/transcribe-denoise-output/"


def denoise(filepath):
    os.makedirs(DENOISE_INPUT, exist_ok=True)
    os.makedirs(DENOISE_OUTPUT, exist_ok=True)

    audio_bytes = normalize_audio(filepath)
    with open(f"{DENOISE_INPUT}/file.wav", "wb") as f:
        f.write(audio_bytes)

    enhance.enhance(DenoiseArgs(), local_out_dir=DENOISE_OUTPUT)

    print("denoise done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("input_file", nargs=1, help="Audio file to transcribe")
    parser.add_argument('--skip-denoise', dest="skip_denoise", help="Skip de-noising", action="store_true")
    args = parser.parse_args()

    INPUT_FILE = args.input_file[0]

    MODEL = "facebook/wav2vec2-large-960h"
    # MODEL = "facebook/hubert-large-ls960-ft"
    DENOISE_MODEL = "master64"

    DENOISE = not args.skip_denoise
    # DENOISE = False

    # if os.path.exists(filepath + ".s2meta"):
    #     exit(0)
    engine = SpeechToTextEngine(MODEL)

    start = time.time()

    if DENOISE:
        denoise(INPUT_FILE)
        result = engine.run(f"{DENOISE_OUTPUT}/file_enhanced.wav")
        os.remove(f"{DENOISE_OUTPUT}/file_enhanced.wav")
    else:
        result = engine.run(INPUT_FILE)

    print(f"Took {time.time() - start:.2f}s")
    print(result)
    # with open(INPUT_FILE + ".s2meta", "w") as f:
    #     f.write(json.dumps({
    #         "content": result,
    #         "_transcribed_by": MODEL,
    #         "_denoised_by": DENOISE_MODEL,
    #     }))
