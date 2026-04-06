import os
import json
import math
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import librosa
import torch
from torch import nn


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_MELS = 80
BEAM_WIDTH = 10

CHARS = list("abcdefghijklmnopqrstuvwxyz '")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
BLANK_LABEL = 0
VOCAB_SIZE = len(CHARS) + 1


def indices_to_text(indices: List[int]) -> str:
    return "".join(IDX_TO_CHAR[i] for i in indices if i in IDX_TO_CHAR)


class SpeechModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d((2, 1))
        )

        self.lstm_input_size = 128 * 10

        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, N_MELS, T)
        x = self.cnn(x)

        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, c * f)

        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.log_softmax(dim=2)
        return x


class SpeechToTextInterface:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        beam_width: int = BEAM_WIDTH,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.beam_width = beam_width

        self.checkpoint = self._load_checkpoint(checkpoint_path)

        self.model = SpeechModel()
        self._load_model_state(self.model, self.checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> Dict:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dict / state_dict.")
        return checkpoint

    @staticmethod
    def _load_model_state(model: nn.Module, checkpoint: Dict) -> None:
        state = checkpoint.get("model_state", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)

        if missing:
            print(f"[Warning] Missing keys while loading model: {len(missing)}")
        if unexpected:
            print(f"[Warning] Unexpected keys while loading model: {len(unexpected)}")

    @staticmethod
    def _normalize_spec(spec: np.ndarray) -> np.ndarray:
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + 1e-5)

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        waveform, sr = sf.read(audio_path)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(np.float32)

        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=400,
            win_length=400,
            hop_length=160,
            power=2.0,
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = self._normalize_spec(spec).astype(np.float32)

        return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, N_MELS, T)

    @staticmethod
    def greedy_decode(log_probs: torch.Tensor, blank: int = BLANK_LABEL) -> List[str]:
        pred_ids = torch.argmax(log_probs, dim=2)
        results = []

        for seq in pred_ids:
            decoded = []
            prev = blank
            for idx in seq.tolist():
                if idx != blank and idx != prev:
                    decoded.append(idx)
                prev = idx
            results.append(indices_to_text(decoded))

        return results

    @staticmethod
    def _log_sum_exp(a: float, b: float) -> float:
        if a == -float("inf"):
            return b
        if b == -float("inf"):
            return a
        if a > b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))

    @classmethod
    def _ctc_beam_search_single(cls, log_probs: torch.Tensor, beam_width: int = BEAM_WIDTH, blank: int = BLANK_LABEL) -> str:
        beams = {("", blank): 0.0}

        for t in range(log_probs.size(0)):
            new_beams = {}

            for (prefix, last_char), score in beams.items():
                for c in range(log_probs.size(1)):
                    new_score = score + log_probs[t, c].item()

                    if c == blank:
                        key = (prefix, blank)
                        if key not in new_beams:
                            new_beams[key] = new_score
                        else:
                            new_beams[key] = cls._log_sum_exp(new_beams[key], new_score)
                    else:
                        char = IDX_TO_CHAR.get(c, "")
                        new_prefix = prefix if c == last_char else prefix + char
                        key = (new_prefix, c)
                        if key not in new_beams:
                            new_beams[key] = new_score
                        else:
                            new_beams[key] = cls._log_sum_exp(new_beams[key], new_score)

            sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
            beams = dict(sorted_beams[:beam_width])

        return max(beams.items(), key=lambda x: x[1])[0][0]

    @classmethod
    def beam_decode(cls, log_probs: torch.Tensor, beam_width: int = BEAM_WIDTH, blank: int = BLANK_LABEL) -> List[str]:
        return [
            cls._ctc_beam_search_single(log_probs[i], beam_width=beam_width, blank=blank)
            for i in range(log_probs.size(0))
        ]

    @torch.no_grad()
    def predict(self, audio_path: str, use_beam: bool = True) -> Dict[str, Union[str, None, Dict[str, Union[str, int]]]]:
        x = self.preprocess_audio(audio_path).to(self.device)
        log_probs = self.model(x)

        if use_beam:
            text = self.beam_decode(log_probs.cpu(), beam_width=self.beam_width, blank=BLANK_LABEL)[0]
            decoder_name = "beam"
        else:
            text = self.greedy_decode(log_probs.cpu(), blank=BLANK_LABEL)[0]
            decoder_name = "greedy"

        return {
            "text": text,
            "confidence": None,
            "meta": {
                "decoder": decoder_name,
                "sample_rate": self.sample_rate,
                "n_mels": self.n_mels,
                "beam_width": self.beam_width if use_beam else None,
            }
        }

    def get_model_info(self) -> Dict[str, Union[str, int, None]]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "model_type": "SpeechModel",
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "vocab_size": VOCAB_SIZE,
            "blank_label": BLANK_LABEL,
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "checkpoint_stage": self.checkpoint.get("stage"),
        }


def load_speech_interface(
    checkpoint_path: str,
    device: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    beam_width: int = BEAM_WIDTH,
) -> SpeechToTextInterface:
    return SpeechToTextInterface(
        checkpoint_path=checkpoint_path,
        device=device,
        sample_rate=sample_rate,
        n_mels=n_mels,
        beam_width=beam_width,
    )


def speech_to_text(
    audio_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
    use_beam: bool = True,
) -> Dict[str, Union[str, None, Dict[str, Union[str, int]]]]:
    interface = load_speech_interface(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return interface.predict(audio_path=audio_path, use_beam=use_beam)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speech-to-text inference interface")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--decoder", type=str, default="beam", choices=["beam", "greedy"])
    args = parser.parse_args()

    interface = load_speech_interface(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    print("Model info:")
    print(json.dumps(interface.get_model_info(), indent=2, ensure_ascii=False))
    print()

    result = interface.predict(
        audio_path=args.audio,
        use_beam=(args.decoder == "beam")
    )
    print("Prediction:")
    print(json.dumps(result, indent=2, ensure_ascii=False))