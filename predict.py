from pget import pget_manifest
import re
import torch
import torchaudio
from typing import List
import numpy as np
import tempfile
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, DiT
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
)
from transformers import pipeline
import soundfile as sf
from cog import BasePredictor, Path, Input
from tqdm import tqdm

SPLIT_WORDS = [
    "but",
    "however",
    "nevertheless",
    "yet",
    "still",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "moreover",
    "furthermore",
    "additionally",
    "meanwhile",
    "alternatively",
    "otherwise",
    "namely",
    "specifically",
    "for example",
    "such as",
    "in fact",
    "indeed",
    "notably",
    "in contrast",
    "on the other hand",
    "conversely",
    "in conclusion",
    "to summarize",
    "finally",
]


def split_text_into_batches(text, max_chars=200, split_words=SPLIT_WORDS):
    if len(text.encode("utf-8")) <= max_chars:
        return [text]
    if text[-1] not in ["。", ".", "!", "！", "?", "？"]:
        text += "."

    sentences = re.split("([。.!?！？])", text)
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]

    batches = []
    current_batch = ""

    def split_by_words(text):
        words = text.split()
        current_word_part = ""
        word_batches = []
        for word in words:
            if (
                len(current_word_part.encode("utf-8")) + len(word.encode("utf-8")) + 1
                <= max_chars
            ):
                current_word_part += word + " "
            else:
                if current_word_part:
                    # Try to find a suitable split word
                    for split_word in split_words:
                        split_index = current_word_part.rfind(" " + split_word + " ")
                        if split_index != -1:
                            word_batches.append(current_word_part[:split_index].strip())
                            current_word_part = (
                                current_word_part[split_index:].strip() + " "
                            )
                            break
                    else:
                        # If no suitable split word found, just append the current part
                        word_batches.append(current_word_part.strip())
                        current_word_part = ""
                current_word_part += word + " "
        if current_word_part:
            word_batches.append(current_word_part.strip())
        return word_batches

    for sentence in sentences:
        if (
            len(current_batch.encode("utf-8")) + len(sentence.encode("utf-8"))
            <= max_chars
        ):
            current_batch += sentence
        else:
            # If adding this sentence would exceed the limit
            if current_batch:
                batches.append(current_batch)
                current_batch = ""

            # If the sentence itself is longer than max_chars, split it
            if len(sentence.encode("utf-8")) > max_chars:
                # First, try to split by colon
                colon_parts = sentence.split(":")
                if len(colon_parts) > 1:
                    for part in colon_parts:
                        if len(part.encode("utf-8")) <= max_chars:
                            batches.append(part)
                        else:
                            # If colon part is still too long, split by comma
                            comma_parts = re.split("[,，]", part)
                            if len(comma_parts) > 1:
                                current_comma_part = ""
                                for comma_part in comma_parts:
                                    if (
                                        len(current_comma_part.encode("utf-8"))
                                        + len(comma_part.encode("utf-8"))
                                        <= max_chars
                                    ):
                                        current_comma_part += comma_part + ","
                                    else:
                                        if current_comma_part:
                                            batches.append(
                                                current_comma_part.rstrip(",")
                                            )
                                        current_comma_part = comma_part + ","
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(","))
                            else:
                                # If no comma, split by words
                                batches.extend(split_by_words(part))
                else:
                    # If no colon, split by comma
                    comma_parts = re.split("[,，]", sentence)
                    if len(comma_parts) > 1:
                        current_comma_part = ""
                        for comma_part in comma_parts:
                            if (
                                len(current_comma_part.encode("utf-8"))
                                + len(comma_part.encode("utf-8"))
                                <= max_chars
                            ):
                                current_comma_part += comma_part + ","
                            else:
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(","))
                                current_comma_part = comma_part + ","
                        if current_comma_part:
                            batches.append(current_comma_part.rstrip(","))
                    else:
                        # If no comma, split by words
                        batches.extend(split_by_words(sentence))
            else:
                current_batch = sentence

    if current_batch:
        batches.append(current_batch)

    return batches


class Predictor(BasePredictor):
    def setup(self):
        pget_manifest('manifest.pget')
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        print(f"Backend: {self.device}")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device=self.device,
        )

        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.nfe_step = 32  # 16, 32
        self.cfg_strength = 2.0
        self.ode_method = "euler"
        self.sway_sampling_coef = -1.0
        self.speed = 1.0
        # self.fix_duration = 27  # None or float (duration in seconds)
        self.fix_duration = None
        ckpt_path = "./ckpts/F5TTS_Base/model_1200000.safetensors"
        vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
        F5TTS_model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
        self.model = CFM(
            transformer=DiT(
                **F5TTS_model_cfg,
                text_num_embeds=vocab_size,
                mel_dim=self.n_mel_channels,
            ),
            mel_spec_kwargs=dict(
                target_sample_rate=self.target_sample_rate,
                n_mel_channels=self.n_mel_channels,
                hop_length=self.hop_length,
            ),
            odeint_kwargs=dict(
                method=self.ode_method,
            ),
            vocab_char_map=vocab_char_map,
        ).to(self.device)
        self.model = load_checkpoint(self.model, ckpt_path, self.device, use_ema=True)

    def _predict(self,
        ref_audio: Path,
        ref_text: str,
        gen_text_batches: List[str],
        remove_silence: bool,
    ):
        audio, sr = torchaudio.load(ref_audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms:
            audio = audio * self.target_rms / rms
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(self.device)
        generated_waves = []
        spectrograms = []
        for i, gen_text in enumerate(tqdm(gen_text_batches)):
            if len(ref_text[-1].encode("utf-8")) == 1:
                ref_text = ref_text + " "
            text_list = [ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)

            ref_audio_len = audio.shape[-1] // self.hop_length
            zh_pause_punc = r"。，、；：？！"
            ref_text_len = len(ref_text.encode("utf-8")) + 3 * len(
                re.findall(zh_pause_punc, ref_text)
            )
            gen_text_len = len(gen_text.encode("utf-8")) + 3 * len(
                re.findall(zh_pause_punc, gen_text)
            )
            duration = ref_audio_len + int(
                ref_audio_len / ref_text_len * gen_text_len / self.speed
            )

            with torch.inference_mode():
                generated, _ = self.model.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=self.nfe_step,
                    cfg_strength=self.cfg_strength,
                    sway_sampling_coef=self.sway_sampling_coef,
                )

            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")

            vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            generated_wave = vocos.decode(generated_mel_spec.cpu())
            if rms < self.target_rms:
                generated_wave = generated_wave * rms / self.target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

        final_wave = np.concatenate(generated_waves)

        if remove_silence:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, final_wave, self.target_sample_rate)
                aseg = AudioSegment.from_file(f.name)
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    non_silent_wave += non_silent_seg
                aseg = non_silent_wave
                aseg.export(f.name, format="wav")
                final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()

        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            sf.write(f.name, final_wave, self.target_sample_rate)
            return Path(f.name)

    def predict(self,
        gen_text: str = Input(description="Text to Generate"),
        ref_audio: Path = Input(description="Reference audio for voice cloning"),
        ref_text: str = Input(description="Reference Text", default=None),
        remove_silence: bool = Input(description="Automatically remove silences?", default=True),
        custom_split_words: str = Input(description="Custom split words, comma separated", default=""),
    ) -> Path:
        if not custom_split_words.strip():
            custom_words = [word.strip() for word in custom_split_words.split(",")]
            global SPLIT_WORDS
            SPLIT_WORDS = custom_words

        print("Generating:", gen_text)
        print("[*] Converting reference audio...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio)

            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave

            audio_duration = len(aseg)
            if audio_duration > 15000:
                print("[!] Audio is over 15s, clipping to only first 15s.")
                aseg = aseg[:15000]
            aseg.export(f.name, format="wav")
            ref_audio = f.name
            print("[+] Converted reference audio.")

        if not ref_text.strip():
            print("[*] No reference text provided, transcribing reference audio...")
            ref_text = self.pipe(
                ref_audio,
                chunk_length_s=30,
                batch_size=128,
                generate_kwargs={"task": "transcribe"},
                return_timestamps=False,
            )["text"].strip()
            print("[+] Finished transcription")
        else:
            print("[*] Using custom reference text...")

        print("[+] Reference text:", ref_text)

        # Split the input text into batches
        print("[*] Forming batches...")
        if len(ref_text.encode("utf-8")) == len(ref_text) and len(
            gen_text.encode("utf-8")
        ) == len(gen_text):
            max_chars = 400 - len(ref_text.encode("utf-8"))
        else:
            max_chars = 300 - len(ref_text.encode("utf-8"))

        gen_text_batches = split_text_into_batches(gen_text, max_chars=max_chars)
        print("[+] Formed batches:", len(gen_text_batches))
        for i, gen_text in enumerate(gen_text_batches):
            print("------ Batch {i+1} -------------------")
            print(gen_text)
            print("--------------------------------------")

        return self._predict(ref_audio, ref_text, gen_text_batches, remove_silence)
