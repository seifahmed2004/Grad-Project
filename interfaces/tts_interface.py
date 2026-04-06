import os
import re
import json
from typing import Dict, Optional, Union

import numpy as np
import soundfile as sf
import librosa
from unidecode import unidecode

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio
SAMPLE_RATE = 22050
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 0
FMAX = 8000

# Model dimensions
EMBED_DIM = 256
ENC_HIDDEN = 256
DEC_HIDDEN = 512
ATTN_DIM = 128
LOCATION_FILTERS = 32
LOCATION_KERNEL = 31
PRENET_DIM = 256
POSTNET_CHANNELS = 512
POSTNET_KERNEL = 5
REDUCTION_FACTOR = 2

# Inference defaults
INFER_STOP_THRESHOLD = 0.55
INFER_MIN_STEPS = 50
INFER_MAX_STEPS = 900

# Text vocabulary
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'-\"()"
PAD_SYM = "<pad>"
SOS_SYM = "<sos>"
EOS_SYM = "<eos>"

_vocab = [PAD_SYM, SOS_SYM, EOS_SYM] + list(_CHARS)
_stoi = {s: i for i, s in enumerate(_vocab)}
_itos = {i: s for s, i in _stoi.items()}

PAD_ID = _stoi[PAD_SYM]
SOS_ID = _stoi[SOS_SYM]
EOS_ID = _stoi[EOS_SYM]
VOCAB_SIZE = len(_vocab)

_ws_re = re.compile(r"\s+")
_allowed_re = re.compile(r"[^a-z0-9 .,!?;:'\"()\-]+")


def clean_text(text: str) -> str:
    text = unidecode(str(text))
    text = text.lower()
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = _allowed_re.sub(" ", text)
    text = _ws_re.sub(" ", text).strip()
    return text


def text_to_ids(text: str, stoi: Dict[str, int]) -> list[int]:
    text = clean_text(text)
    sos_id = stoi.get(SOS_SYM, SOS_ID)
    eos_id = stoi.get(EOS_SYM, EOS_ID)
    pad_id = stoi.get(PAD_SYM, PAD_ID)
    return [sos_id] + [stoi.get(ch, pad_id) for ch in text] + [eos_id]


def log_mel_to_audio(mel_log: np.ndarray, n_iter: int = 32) -> np.ndarray:
    mel_log = mel_log.astype(np.float32)

    # safety clamp
    mel_log = np.nan_to_num(mel_log, nan=-10.0, posinf=2.0, neginf=-10.0)
    mel_log = np.clip(mel_log, -10.0, 2.0)

    mel = np.exp(mel_log.T).astype(np.float32)  # [80, T]

    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        power=1.0,
        n_iter=n_iter,
        dtype=np.float32,
    )
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=(256, 256), dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim if i == 0 else sizes[i - 1], sizes[i])
            for i in range(len(sizes))
        ])
        self.dp = dropout
        

    def forward(self, x):
        for lin in self.layers:
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dp, training=True)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            ) for _ in range(3)
        ])
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        x = self.embedding(x).transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1)
        )
        return out


class LocationLayer(nn.Module):
    def __init__(self, n_filters, kernel_size, attn_dim):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, n_filters, kernel_size=kernel_size, padding=pad, bias=False)
        self.dense = nn.Linear(n_filters, attn_dim, bias=False)

    def forward(self, attn_cat):
        return self.dense(self.conv(attn_cat).transpose(1, 2))


class LocationSensitiveAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.query_layer = nn.Linear(dec_dim, attn_dim, bias=False)
        self.memory_layer = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=True)
        self.location_layer = LocationLayer(LOCATION_FILTERS, LOCATION_KERNEL, attn_dim)

    def forward(self, query, memory, processed_memory, attn_cat, mask):
        pq = self.query_layer(query).unsqueeze(1)
        pl = self.location_layer(attn_cat)
        energy = self.v(torch.tanh(pq + pl + processed_memory)).squeeze(-1)
        if mask is not None:
            energy.data.masked_fill_(~mask, -float("inf"))
        weights = F.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return weights, context


class Postnet(nn.Module):
    def __init__(self, n_mels=80, channels=512, kernel=5, n_convs=5, dropout=0.5):
        super().__init__()
        layers = []
        in_ch = n_mels
        for i in range(n_convs):
            out_ch = n_mels if i == n_convs - 1 else channels
            act = nn.Identity() if i == n_convs - 1 else nn.Tanh()
            layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                act,
                nn.Dropout(dropout),
            ))
            in_ch = out_ch
        self.convs = nn.ModuleList(layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        return x.transpose(1, 2)


class TacotronTTS(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int = 0):
        super().__init__()
        enc_out_dim = 2 * ENC_HIDDEN

        self.encoder = Encoder(vocab_size, EMBED_DIM, ENC_HIDDEN, pad_id=pad_id)
        self.prenet = Prenet(N_MELS, sizes=(PRENET_DIM, PRENET_DIM))
        self.attention_rnn = nn.LSTMCell(PRENET_DIM + enc_out_dim, DEC_HIDDEN)
        self.attention = LocationSensitiveAttention(enc_out_dim, DEC_HIDDEN, ATTN_DIM)
        self.decoder_rnn = nn.LSTMCell(DEC_HIDDEN + enc_out_dim, DEC_HIDDEN)
        self.mel_proj = nn.Linear(DEC_HIDDEN + enc_out_dim, N_MELS * REDUCTION_FACTOR)
        self.stop_proj = nn.Linear(DEC_HIDDEN + enc_out_dim, 1)
        self.postnet = Postnet(N_MELS, POSTNET_CHANNELS, POSTNET_KERNEL)

    def _init_states(self, batch_size, t_enc, memory, device):
        h_att = torch.zeros(batch_size, DEC_HIDDEN, device=device)
        c_att = torch.zeros(batch_size, DEC_HIDDEN, device=device)
        h_dec = torch.zeros(batch_size, DEC_HIDDEN, device=device)
        c_dec = torch.zeros(batch_size, DEC_HIDDEN, device=device)
        attn_w = torch.zeros(batch_size, t_enc, device=device)
        attn_w_cum = torch.zeros(batch_size, t_enc, device=device)
        context = torch.zeros(batch_size, memory.size(2), device=device)
        return h_att, c_att, h_dec, c_dec, attn_w, attn_w_cum, context

    def forward(
        self,
        text_ids,
        text_lens,
        target_mels=None,
        teacher_forcing_ratio=0.0,
        max_decoder_steps=INFER_MAX_STEPS,
        stop_threshold=INFER_STOP_THRESHOLD,
        min_decoder_steps=INFER_MIN_STEPS,
    ):
        batch_size = text_ids.size(0)
        device = text_ids.device

        memory = self.encoder(text_ids, text_lens)
        processed_memory = self.attention.memory_layer(memory)
        t_enc = memory.size(1)

        mask = torch.arange(t_enc, device=device).unsqueeze(0) < text_lens.unsqueeze(1)

        h_att, c_att, h_dec, c_dec, attn_w, attn_w_cum, context = \
            self._init_states(batch_size, t_enc, memory, device)

        n_steps = max_decoder_steps if target_mels is None else target_mels.size(1) // REDUCTION_FACTOR

        mel_outputs = []
        stop_outputs = []
        attn_outputs = []

        dec_input = torch.zeros(batch_size, N_MELS, device=device)

        for step in range(n_steps):
            prenet_out = self.prenet(dec_input)

            attn_rnn_input = torch.cat([prenet_out, context], dim=-1)
            h_att, c_att = self.attention_rnn(attn_rnn_input, (h_att, c_att))

            attn_cat = torch.stack([attn_w, attn_w_cum], dim=1)
            attn_w, context = self.attention(h_att, memory, processed_memory, attn_cat, mask)
            attn_w_cum = attn_w_cum + attn_w

            dec_rnn_input = torch.cat([h_att, context], dim=-1)
            h_dec, c_dec = self.decoder_rnn(dec_rnn_input, (h_dec, c_dec))

            dec_out = torch.cat([h_dec, context], dim=-1)
            mel_frame = self.mel_proj(dec_out)
            stop_logit = self.stop_proj(dec_out)

            mel_frame = mel_frame.view(batch_size, REDUCTION_FACTOR, N_MELS)

            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_logit.squeeze(-1))
            attn_outputs.append(attn_w)

            if target_mels is not None:
                t_mel = target_mels.size(1)
                if torch.rand(1).item() < teacher_forcing_ratio:
                    frame_idx = min((step + 1) * REDUCTION_FACTOR - 1, t_mel - 1)
                    dec_input = target_mels[:, frame_idx, :]
                else:
                    dec_input = mel_frame[:, -1, :]
            else:
                dec_input = mel_frame[:, -1, :]
                if step >= min_decoder_steps:
                    stop_prob = torch.sigmoid(stop_logit.squeeze(-1))
                    if (stop_prob > stop_threshold).all():
                        break

        mel_pred = torch.cat(mel_outputs, dim=1)
        stop_logits = torch.stack(stop_outputs, dim=1)
        attn = torch.stack(attn_outputs, dim=1)
        mel_post = mel_pred + self.postnet(mel_pred)

        return mel_pred, mel_post, stop_logits, attn


class TextToSpeechInterface:
    """
    Integration-ready TTS interface.

    Main usage:
        interface = TextToSpeechInterface("best.pt")
        result = interface.synthesize("hello world", output_audio_path="out.wav")
        print(result["output_audio_path"])
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        stop_threshold: float = INFER_STOP_THRESHOLD,
        min_steps: int = INFER_MIN_STEPS,
        max_steps: int = INFER_MAX_STEPS,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.stop_threshold = stop_threshold
        self.min_steps = min_steps
        self.max_steps = max_steps

        self.checkpoint = self._load_checkpoint(checkpoint_path)
        self.vocab = self.checkpoint.get("vocab", _vocab)
        self.stoi = self.checkpoint.get("stoi", _stoi)

        self.pad_id = self.stoi.get(PAD_SYM, PAD_ID)
        self.vocab_size = len(self.vocab)

        self.model = TacotronTTS(vocab_size=self.vocab_size, pad_id=self.pad_id)
        self._load_model_state(self.model, self.checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> Dict:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dict / full checkpoint.")
        return checkpoint

    @staticmethod
    def _load_model_state(model: nn.Module, checkpoint: Dict) -> None:
        state = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Warning] Missing keys while loading model: {len(missing)}")
        if unexpected:
            print(f"[Warning] Unexpected keys while loading model: {len(unexpected)}")

    def preprocess_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        cleaned = clean_text(text)
        ids = text_to_ids(cleaned, self.stoi)
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)
        return ids_t, lengths, cleaned

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        output_audio_path: Optional[str] = None,
        griffin_lim_iters: int = 32,
    ) -> Dict[str, Union[str, Dict, int, float]]:
        ids_t, lengths, cleaned = self.preprocess_text(text)

        _, mel_post, stop_logits, attn = self.model(
            ids_t,
            lengths,
            target_mels=None,
            teacher_forcing_ratio=0.0,
            max_decoder_steps=min(self.max_steps, 180),
            stop_threshold=max(self.stop_threshold, 0.7),
            min_decoder_steps=min(self.min_steps, 20),
        )

        mel = mel_post[0].float().cpu().numpy()

        # hard safety limit on mel length
        max_mel_frames = 220
        if mel.shape[0] > max_mel_frames:
            mel = mel[:max_mel_frames]

        audio = log_mel_to_audio(mel, n_iter=griffin_lim_iters)

        if output_audio_path is not None:
            out_dir = os.path.dirname(output_audio_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            sf.write(output_audio_path, audio, SAMPLE_RATE)

        return {
            "text": cleaned,
            "output_audio_path": output_audio_path,
            "meta": {
                "sample_rate": SAMPLE_RATE,
                "mel_frames": int(mel.shape[0]),
                "mel_bins": int(mel.shape[1]),
                "audio_num_samples": int(len(audio)),
                "audio_duration_sec": round(len(audio) / SAMPLE_RATE, 2),
                "griffin_lim_iters": griffin_lim_iters,
                "stop_threshold": max(self.stop_threshold, 0.7),
                "min_steps": min(self.min_steps, 20),
                "max_steps": min(self.max_steps, 180),
                "attention_shape": list(attn[0].shape),
                "stop_shape": list(stop_logits.shape),
            }
        }

    def get_model_info(self) -> Dict[str, Union[str, int, float, None]]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "model_type": "TacotronTTS",
            "vocab_size": self.vocab_size,
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "reduction_factor": REDUCTION_FACTOR,
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "best_val": self.checkpoint.get("best_val"),
        }


def load_tts_interface(
    checkpoint_path: str,
    device: Optional[str] = None,
    stop_threshold: float = INFER_STOP_THRESHOLD,
    min_steps: int = INFER_MIN_STEPS,
    max_steps: int = INFER_MAX_STEPS,
) -> TextToSpeechInterface:
    return TextToSpeechInterface(
        checkpoint_path=checkpoint_path,
        device=device,
        stop_threshold=stop_threshold,
        min_steps=min_steps,
        max_steps=max_steps,
    )


def text_to_speech(
    text: str,
    checkpoint_path: str,
    output_audio_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Union[str, Dict, int, float]]:
    interface = load_tts_interface(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return interface.synthesize(
        text=text,
        output_audio_path=output_audio_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-speech inference interface")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--output", type=str, default="outputs/generated.wav", help="Output wav path")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    interface = load_tts_interface(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    print("Model info:")
    print(json.dumps(interface.get_model_info(), indent=2, ensure_ascii=False))
    print()

    result = interface.synthesize(
        text=args.text,
        output_audio_path=args.output,
    )
    print("Synthesis result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))