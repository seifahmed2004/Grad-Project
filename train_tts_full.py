#!/usr/bin/env python3
"""
Full From-Scratch TTS Pipeline for HPC
=======================================
Stage 1: Tacotron2-style acoustic model  (text → mel spectrogram)
Stage 2: HiFi-GAN vocoder               (mel spectrogram → waveform)

Run examples
------------
# Train acoustic model from scratch (or resume automatically):
python train_tts_full.py --stage acoustic \
    --data-dir /path/to/LJSpeech-1.1 \
    --work-dir /path/to/project

# Train vocoder after acoustic model is done:
python train_tts_full.py --stage vocoder \
    --data-dir /path/to/LJSpeech-1.1 \
    --work-dir /path/to/project

# Synthesize speech from a trained full pipeline:
python train_tts_full.py --stage synthesize \
    --work-dir /path/to/project \
    --text "Hello, this is my graduation project."

Everything is from scratch — no pretrained weights, no pretrained models.
Checkpoints saved every 10 epochs + best model always kept.
Auto-resumes from latest checkpoint on every run.
"""

import os, re, math, time, random, warnings, wave, contextlib, argparse, json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unidecode import unidecode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="From-Scratch TTS — HPC version")
    p.add_argument("--stage", choices=["acoustic", "vocoder", "synthesize"],
                   default="acoustic", help="Which stage to run")
    p.add_argument("--data-dir",  type=str, default="./LJSpeech-1.1")
    p.add_argument("--work-dir",  type=str, default="./tts_project")
    p.add_argument("--epochs",    type=int, default=None,
                   help="Override total epochs (default: 250 acoustic, 300 vocoder)")
    p.add_argument("--batch-size",type=int, default=None,
                   help="Override batch size")
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--text",      type=str,
                   default="Hello, this is my graduation project text to speech model, "
                           "built completely from scratch.")
    p.add_argument("--resume",    action="store_true", default=True,
                   help="Resume from checkpoint (default: True)")
    p.add_argument("--no-amp",    action="store_true",
                   help="Disable automatic mixed precision")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED AUDIO CONFIG  (both stages use the same mel settings)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 22050
N_FFT        = 1024
WIN_LENGTH   = 1024
HOP_LENGTH   = 256
N_MELS       = 80
FMIN         = 0
FMAX         = 8000

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_wav(path: str) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.clip(wav, -1.0, 1.0)


def wav_to_log_mel(wav: np.ndarray) -> np.ndarray:
    """Returns [T, N_MELS] float32 log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=1.0,
    )
    return np.log(np.maximum(mel, 1e-5)).astype(np.float32).T   # [T, 80]


def log_mel_to_audio_griffinlim(mel_log: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """Griffin-Lim fallback vocoder (used only if HiFi-GAN not available)."""
    mel = np.exp(mel_log.astype(np.float32).T)
    audio = librosa.feature.inverse.mel_to_audio(
        mel, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, fmin=FMIN, fmax=FMAX,
        power=1.0, n_iter=n_iter,
    )
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'-\"()"
PAD_SYM, SOS_SYM, EOS_SYM = "<pad>", "<sos>", "<eos>"
vocab  = [PAD_SYM, SOS_SYM, EOS_SYM] + list(_CHARS)
stoi   = {s: i for i, s in enumerate(vocab)}
itos   = {i: s for s, i in stoi.items()}
PAD_ID = stoi[PAD_SYM]
SOS_ID = stoi[SOS_SYM]
EOS_ID = stoi[EOS_SYM]
VOCAB_SIZE = len(vocab)

_ws_re      = re.compile(r"\s+")
_allowed_re = re.compile(r"[^a-z0-9 .,!?;:'\"()\-]+")

def clean_text(text: str) -> str:
    text = unidecode(str(text)).lower()
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = _allowed_re.sub(" ", text)
    return _ws_re.sub(" ", text).strip()

def text_to_ids(text: str):
    return [SOS_ID] + [stoi.get(ch, PAD_ID) for ch in clean_text(text)] + [EOS_ID]


# ─────────────────────────────────────────────────────────────────────────────
# METADATA LOADING (shared)
# ─────────────────────────────────────────────────────────────────────────────
def load_metadata(data_dir: Path,
                  min_text=5, max_text=180,
                  min_mel=20, max_mel=870,
                  val_size=300):
    csv_path  = data_dir / "metadata.csv"
    wavs_dir  = data_dir / "wavs"

    def mel_len(wav_path):
        with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
            return math.ceil(wf.getnframes() / HOP_LENGTH)

    records, dropped = [], 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2: continue
            utt_id    = parts[0]
            norm_text = parts[2] if len(parts) > 2 else parts[1]
            wav_path  = wavs_dir / f"{utt_id}.wav"
            if not wav_path.exists(): dropped += 1; continue
            text = clean_text(norm_text)
            if not (min_text <= len(text) <= max_text): dropped += 1; continue
            try:   ml = mel_len(wav_path)
            except Exception: dropped += 1; continue
            if not (min_mel <= ml <= max_mel): dropped += 1; continue
            records.append({"id": utt_id, "text": text,
                            "wav_path": str(wav_path), "mel_len": ml})

    random.seed(SEED); random.shuffle(records)
    vs = min(val_size, max(100, len(records) // 20))
    print(f"  Total: {len(records)}  Train: {len(records)-vs}  "
          f"Val: {vs}  Dropped: {dropped}")
    return records[vs:], records[:vs]


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — ACOUSTIC MODEL  (Tacotron 2 style)
# ═════════════════════════════════════════════════════════════════════════════

# ── Acoustic config ──────────────────────────────────────────────────────────
AC = dict(
    epochs          = 250,
    batch_size      = 32,
    lr              = 1e-3,
    lr_min          = 5e-5,
    weight_decay    = 1e-6,
    grad_clip       = 1.0,
    patience        = 25,
    sched_patience  = 4,

    embed_dim       = 256,
    enc_hidden      = 256,
    dec_hidden      = 512,
    attn_dim        = 128,
    loc_filters     = 32,
    loc_kernel      = 31,
    prenet_dim      = 256,
    postnet_ch      = 512,
    postnet_kernel  = 5,
    reduction       = 2,

    stop_weight     = 8.0,
    tf_hold_frac    = 0.40,
    tf_start        = 1.0,
    tf_end          = 0.85,
    ga_hold_frac    = 0.30,
    ga_max          = 1.0,
    ga_min          = 0.05,
    ga_g            = 0.2,

    infer_stop_thr  = 0.55,
    infer_min_steps = 50,
    infer_max_steps = 900,
    ckpt_every      = 10,       # save numbered checkpoint every N epochs
)


# ── Acoustic dataset ─────────────────────────────────────────────────────────
class AcousticDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        ids = np.array(text_to_ids(rec["text"]), dtype=np.int64)
        mel = wav_to_log_mel(load_wav(rec["wav_path"]))
        return {"ids": ids, "mel": mel}


def acoustic_collate(batch):
    batch = sorted(batch, key=lambda x: len(x["ids"]), reverse=True)
    il    = [len(b["ids"])        for b in batch]
    ml    = [b["mel"].shape[0]    for b in batch]
    max_il = max(il)
    max_ml = max(ml)
    R = AC["reduction"]
    if max_ml % R != 0:
        max_ml += R - (max_ml % R)
    B = len(batch)
    ids_t  = torch.full((B, max_il), PAD_ID, dtype=torch.long)
    mels_t = torch.zeros((B, max_ml, N_MELS))
    stop_t = torch.zeros((B, max_ml // R))
    for i, b in enumerate(batch):
        ids_t[i, :il[i]]  = torch.tensor(b["ids"])
        mels_t[i, :ml[i]] = torch.tensor(b["mel"])
        stop_t[i, math.ceil(ml[i] / R) - 1:] = 1.0
    return (ids_t,
            torch.tensor(il, dtype=torch.long),
            mels_t,
            torch.tensor(ml, dtype=torch.long),
            stop_t)


# ── Acoustic model layers ────────────────────────────────────────────────────
class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=(256, 256), p=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim if i == 0 else sizes[i-1], sizes[i])
            for i in range(len(sizes))])
        self.p = p

    def forward(self, x):
        for lin in self.layers:
            x = F.dropout(F.relu(lin(x)), p=self.p, training=True)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = AC["embed_dim"]
        h = AC["enc_hidden"]
        self.embed = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d, d, 5, padding=2),
                nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(0.5))
            for _ in range(3)])
        self.bilstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        x = self.embed(x).transpose(1, 2)
        for c in self.convs: x = c(x)
        x = x.transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))
        return out   # [B, T, 2*enc_hidden]


class LocationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        k = AC["loc_kernel"]
        self.conv  = nn.Conv1d(2, AC["loc_filters"], k, padding=(k-1)//2, bias=False)
        self.dense = nn.Linear(AC["loc_filters"], AC["attn_dim"], bias=False)

    def forward(self, x):
        return self.dense(self.conv(x).transpose(1, 2))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        enc = 2 * AC["enc_hidden"]
        self.Q   = nn.Linear(AC["dec_hidden"], AC["attn_dim"], bias=False)
        self.M   = nn.Linear(enc,              AC["attn_dim"], bias=False)
        self.v   = nn.Linear(AC["attn_dim"], 1, bias=True)
        self.loc = LocationLayer()

    def forward(self, q, mem, pmem, attn_cat, mask):
        e = self.v(torch.tanh(self.Q(q).unsqueeze(1) +
                              self.loc(attn_cat) + pmem)).squeeze(-1)
        if mask is not None:
            e.data.masked_fill_(~mask, -float("inf"))
        w = F.softmax(e, dim=1)
        c = torch.bmm(w.unsqueeze(1), mem).squeeze(1)
        return w, c


class Postnet(nn.Module):
    def __init__(self):
        super().__init__()
        k  = AC["postnet_kernel"]
        ch = AC["postnet_ch"]
        layers, in_ch = [], N_MELS
        for i in range(5):
            out_ch = N_MELS if i == 4 else ch
            layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.Identity() if i == 4 else nn.Tanh(),
                nn.Dropout(0.5)))
            in_ch = out_ch
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        for l in self.net: x = l(x)
        return x.transpose(1, 2)


class AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        enc = 2 * AC["enc_hidden"]
        D   = AC["dec_hidden"]
        P   = AC["prenet_dim"]
        R   = AC["reduction"]
        self.encoder  = Encoder()
        self.prenet   = Prenet(N_MELS, (P, P))
        self.attn_rnn = nn.LSTMCell(P + enc, D)
        self.attn     = Attention()
        self.dec_rnn  = nn.LSTMCell(D + enc, D)
        self.mel_proj = nn.Linear(D + enc, N_MELS * R)
        self.stp_proj = nn.Linear(D + enc, 1)
        self.postnet  = Postnet()

    def forward(self, ids, lens, target=None, tf=1.0,
                max_steps=900, stop_thr=0.55, min_steps=50):
        B, dev = ids.size(0), ids.device
        enc    = 2 * AC["enc_hidden"]
        D      = AC["dec_hidden"]
        R      = AC["reduction"]

        mem    = self.encoder(ids, lens)           # [B, T_enc, enc]
        pmem   = self.attn.M(mem)                  # [B, T_enc, attn_dim]
        T_enc  = mem.size(1)
        mask   = torch.arange(T_enc, device=dev).unsqueeze(0) < lens.unsqueeze(1)

        ha = ca = hd = cd = torch.zeros(B, D, device=dev)
        aw = aw_cum = torch.zeros(B, T_enc, device=dev)
        ctx   = torch.zeros(B, enc, device=dev)
        frame = torch.zeros(B, N_MELS, device=dev)

        n   = (target.size(1) // R) if target is not None else max_steps
        T_t = target.size(1) if target is not None else None

        mel_out, stp_out, attn_out = [], [], []

        for step in range(n):
            ha, ca = self.attn_rnn(
                torch.cat([self.prenet(frame), ctx], -1), (ha, ca))
            aw, ctx = self.attn(ha, mem, pmem,
                                torch.stack([aw, aw_cum], 1), mask)
            aw_cum  = aw_cum + aw
            hd, cd  = self.dec_rnn(torch.cat([ha, ctx], -1), (hd, cd))
            out     = torch.cat([hd, ctx], -1)
            mf      = self.mel_proj(out).view(B, R, N_MELS)
            sl      = self.stp_proj(out)
            mel_out.append(mf); stp_out.append(sl.squeeze(-1)); attn_out.append(aw)

            if target is not None:
                frame = (target[:, min((step+1)*R-1, T_t-1), :]
                         if random.random() < tf else mf[:, -1, :])
            else:
                frame = mf[:, -1, :]
                if step >= min_steps and (torch.sigmoid(sl.squeeze(-1)) > stop_thr).all():
                    break

        mel   = torch.cat(mel_out, 1)
        stp   = torch.stack(stp_out, 1)
        attn  = torch.stack(attn_out, 1)
        return mel, mel + self.postnet(mel), stp, attn


# ── Acoustic losses ──────────────────────────────────────────────────────────
def make_mask(lens, T, dev):
    return torch.arange(T, device=dev).unsqueeze(0) < lens.unsqueeze(1)

def masked_l1(pred, tgt, lens):
    T = min(pred.size(1), tgt.size(1))
    m = make_mask(lens.clamp(max=T), T, pred.device).unsqueeze(-1).float()
    return ((pred[:,:T] - tgt[:,:T]).abs() * m).sum() / (m.sum() * N_MELS + 1e-8)

def masked_bce(logits, tgt, lens):
    R  = AC["reduction"]
    rl = torch.ceil(lens.float() / R).long().clamp(max=logits.size(1))
    T  = logits.size(1)
    m  = make_mask(rl, T, logits.device).float()
    pw = torch.tensor(AC["stop_weight"], device=logits.device)
    return (F.binary_cross_entropy_with_logits(
        logits, tgt[:,:T], reduction="none", pos_weight=pw) * m
    ).sum() / (m.sum() + 1e-8)

def guided_attn(attn, il, ml):
    R = AC["reduction"]; g = AC["ga_g"]
    B, Td, Te_max = attn.shape
    total, cnt = 0.0, 0
    for b in range(B):
        Td_ = min(Td, math.ceil(ml[b].item() / R))
        Te_ = min(Te_max, il[b].item())
        if Td_ <= 1 or Te_ <= 1: continue
        t = torch.arange(Td_, device=attn.device, dtype=torch.float32).unsqueeze(1) / Td_
        n = torch.arange(Te_, device=attn.device, dtype=torch.float32).unsqueeze(0) / Te_
        W = 1.0 - torch.exp(-((n - t)**2) / (2*g*g))
        total += (attn[b, :Td_, :Te_] * W).mean(); cnt += 1
    return total / cnt if cnt > 0 else attn.sum() * 0

def diag_score(attn, il, ml):
    R = AC["reduction"]
    B, Td, Te_max = attn.shape; scores = []
    for b in range(min(B, 4)):
        Td_ = min(Td, math.ceil(ml[b].item() / R))
        Te_ = min(Te_max, il[b].item())
        if Td_ <= 1 or Te_ <= 1: continue
        t = torch.arange(Td_, device=attn.device, dtype=torch.float32) / Td_
        n = torch.argmax(attn[b, :Td_, :Te_], dim=1).float() / Te_
        scores.append(1.0 - (n - t).abs().mean().item())
    return float(np.mean(scores)) if scores else 0.0

def get_tf(epoch, total):
    hu = int(total * AC["tf_hold_frac"])
    return AC["tf_start"] if epoch <= hu else \
           AC["tf_start"] + (epoch-hu)/max(total-hu, 1) * (AC["tf_end"]-AC["tf_start"])

def get_ga(epoch, total):
    hu = int(total * AC["ga_hold_frac"])
    return AC["ga_max"] if epoch <= hu else \
           AC["ga_max"] + (epoch-hu)/max(total-hu, 1) * (AC["ga_min"]-AC["ga_max"])


# ── Acoustic synthesize ───────────────────────────────────────────────────────
@torch.no_grad()
def acoustic_synthesize(model, text, device, use_amp=True):
    m = model.module if isinstance(model, nn.DataParallel) else model
    m.eval()
    ids  = torch.tensor([text_to_ids(text)], dtype=torch.long, device=device)
    lens = torch.tensor([ids.size(1)], dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        _, mel, _, attn = m(ids, lens, target=None,
                            max_steps=AC["infer_max_steps"],
                            stop_thr=AC["infer_stop_thr"],
                            min_steps=AC["infer_min_steps"])
    return mel[0].float().cpu().numpy(), attn[0].float().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 TRAINER
# ─────────────────────────────────────────────────────────────────────────────
def train_acoustic(args, device, use_amp):
    work  = Path(args.work_dir)
    ckpts = work / "acoustic" / "checkpoints"
    samps = work / "acoustic" / "samples"
    plots = work / "acoustic" / "plots"
    for d in [ckpts, samps, plots]: d.mkdir(parents=True, exist_ok=True)

    EPOCHS = args.epochs or AC["epochs"]
    BS     = args.batch_size or AC["batch_size"]

    print("=" * 60)
    print("STAGE 1 — ACOUSTIC MODEL TRAINING")
    print(f"  Data  : {args.data_dir}")
    print(f"  Work  : {work}/acoustic")
    print(f"  Epochs: {EPOCHS}   Batch: {BS}   Device: {device}")
    print("=" * 60)

    print("\nLoading metadata...")
    trn, val = load_metadata(Path(args.data_dir))

    trn_loader = DataLoader(AcousticDataset(trn), batch_size=BS, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=acoustic_collate, drop_last=True)
    val_loader = DataLoader(AcousticDataset(val), batch_size=BS, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=acoustic_collate)
    print(f"  Train batches: {len(trn_loader)}  Val batches: {len(val_loader)}")

    model = AcousticModel()
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"  Using DataParallel across {n_gpu} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=AC["lr"],
                               weight_decay=AC["weight_decay"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=AC["sched_patience"], min_lr=AC["lr_min"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # ── Checkpoint helpers ────────────────────────────────────────────────
    def save(path, epoch, best_val, hist):
        m = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({"epoch": epoch, "model": m.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": best_val, "history": hist}, path)
        print(f"  ✓ Checkpoint saved → {Path(path).name}")

    def load(path):
        ck = torch.load(path, map_location='cpu')
        m  = model.module if isinstance(model, nn.DataParallel) else model
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        return ck["epoch"], ck["best_val"], ck.get("history", {
            "train": [], "val": [], "diag": [], "lr": []})

    # ── Auto-resume ────────────────────────────────────────────────────────
    start, best_val = 1, float("inf")
    hist = {"train": [], "val": [], "diag": [], "lr": []}
    no_imp = 0

    latest = ckpts / "latest.pt"
    best   = ckpts / "best.pt"

    if args.resume and latest.exists():
        print(f"\nResuming from: {latest}")
        start, best_val, hist = load(latest)
        print(f"  Epoch {start}  best_val={best_val:.4f}  "
              f"history={len(hist['train'])} epochs")
        start += 1
    else:
        print("\nStarting from scratch (epoch 1)")

    # ── Validation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(epoch):
        model.eval(); tot, n = 0.0, 0; last = None
        for ids, il, mels, ml, stop in val_loader:
            ids=ids.to(device); il=il.to(device)
            mels=mels.to(device); ml=ml.to(device); stop=stop.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                p, pp, sl, attn = model(ids, il, target=mels, tf=1.0)
                loss = (masked_l1(p, mels, ml) + masked_l1(pp, mels, ml)
                        + masked_bce(sl, stop, ml)
                        + get_ga(epoch, EPOCHS) * guided_attn(attn, il, ml))
            tot += loss.item() * ids.size(0); n += ids.size(0)
            last = (attn, il, ml)
        ds = diag_score(*last) if last else 0.0
        return tot / max(n, 1), ds

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nTraining epoch {start} → {EPOCHS}")
    for epoch in range(start, EPOCHS + 1):
        model.train()
        tf = get_tf(epoch, EPOCHS)
        ga = get_ga(epoch, EPOCHS)
        tr_loss, n, t0 = 0.0, 0, time.time()

        for ids, il, mels, ml, stop in trn_loader:
            ids=ids.to(device, non_blocking=True)
            il=il.to(device, non_blocking=True)
            mels=mels.to(device, non_blocking=True)
            ml=ml.to(device, non_blocking=True)
            stop=stop.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                p, pp, sl, attn = model(ids, il, target=mels, tf=tf)
                loss = (masked_l1(p, mels, ml) + masked_l1(pp, mels, ml)
                        + masked_bce(sl, stop, ml)
                        + ga * guided_attn(attn, il, ml))
            if not torch.isfinite(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), AC["grad_clip"])
            scaler.step(opt); scaler.update()
            tr_loss += loss.item() * ids.size(0); n += ids.size(0)

        tr_loss /= max(n, 1)
        vl, ds   = validate(epoch)
        sched.step(vl)
        lr = opt.param_groups[0]["lr"]

        hist["train"].append(tr_loss); hist["val"].append(vl)
        hist["diag"].append(ds);       hist["lr"].append(lr)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train={tr_loss:.4f} Val={vl:.4f} | "
              f"Diag={ds:.3f} | TF={tf:.2f} GA={ga:.3f} | "
              f"LR={lr:.1e} | {elapsed:.0f}s")

        # Always save latest
        save(latest, epoch, best_val, hist)

        # Save numbered checkpoint every N epochs
        if epoch % AC["ckpt_every"] == 0:
            save(ckpts / f"epoch_{epoch:04d}.pt", epoch, best_val, hist)

        # Save best
        if vl < best_val:
            best_val = vl; no_imp = 0
            save(best, epoch, best_val, hist)
            print(f"  ★ New best: {best_val:.4f}")
        else:
            no_imp += 1
            if no_imp >= AC["patience"]:
                print("  Early stopping."); break

        # Periodic audio sample + plot
        if epoch % AC["ckpt_every"] == 0 or epoch == 1:
            try:
                mel, attn_np = acoustic_synthesize(
                    model, args.text, device, use_amp)
                out = str(samps / f"epoch_{epoch:04d}.wav")
                sf.write(out, log_mel_to_audio_griffinlim(mel), SAMPLE_RATE)
                print(f"  Sample: {out}")
                # Attention + mel plot
                fig, ax = plt.subplots(1, 2, figsize=(14, 4))
                ax[0].imshow(attn_np, aspect="auto", origin="lower")
                ax[0].set_title(f"Attention (epoch {epoch})")
                ax[0].set_xlabel("Encoder steps"); ax[0].set_ylabel("Decoder steps")
                ax[1].imshow(mel.T, aspect="auto", origin="lower")
                ax[1].set_title("Predicted Mel"); ax[1].set_xlabel("Time")
                plt.tight_layout()
                plt.savefig(str(plots / f"epoch_{epoch:04d}.png"), dpi=80)
                plt.close()
            except Exception as e:
                print(f"  Sample failed: {e}")

            # Training curve
            if len(hist["train"]) > 1:
                ep = list(range(1, len(hist["train"]) + 1))
                fig, ax = plt.subplots(1, 3, figsize=(18, 4))
                ax[0].plot(ep, hist["train"], label="Train")
                ax[0].plot(ep, hist["val"],   label="Val")
                ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(True)
                ax[1].plot(ep, hist["diag"], color="green")
                ax[1].axhline(0.5, color="orange", ls="--", label="0.5")
                ax[1].axhline(0.7, color="red",    ls="--", label="0.7")
                ax[1].set_ylim(0, 1); ax[1].set_title("Diag Score")
                ax[1].legend(); ax[1].grid(True)
                ax[2].plot(ep, hist["lr"])
                ax[2].set_yscale("log"); ax[2].set_title("LR"); ax[2].grid(True)
                plt.tight_layout()
                plt.savefig(str(work / "acoustic" / "training_curve.png"), dpi=100)
                plt.close()

    print("\nAcoustic model training complete.")
    print(f"  Best checkpoint : {best}")
    print(f"  Audio samples   : {samps}")


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — HiFi-GAN VOCODER  (from scratch, pure PyTorch)
# ═════════════════════════════════════════════════════════════════════════════

# ── Vocoder config ────────────────────────────────────────────────────────────
VC = dict(
    epochs           = 300,
    batch_size       = 16,
    segment_frames   = 64,       # mel frames per training segment
    lr_g             = 2e-4,
    lr_d             = 2e-4,
    lr_decay         = 0.999,    # multiply LR every epoch
    weight_decay     = 0.0,
    grad_clip        = 5.0,
    patience         = 50,
    ckpt_every       = 10,

    # Generator upsampling chain (product must equal HOP_LENGTH=256)
    upsample_rates   = [8, 8, 4],          # 8*8*4 = 256 ✓
    upsample_kernels = [16, 16, 8],
    resblock_kernels = [3, 7, 11],
    resblock_dilations = [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
    gen_init_ch      = 256,

    # Multi-Period Discriminator periods
    mpd_periods      = [2, 3, 5, 7, 11],
    msd_scales       = [1, 2, 4],          # Multi-Scale Discriminator

    lambda_fm        = 2.0,    # feature matching loss weight
    lambda_mel       = 45.0,   # mel reconstruction loss weight
)


# ── HiFi-GAN Generator ───────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """HiFi-GAN residual block with multiple dilation cycles."""
    def __init__(self, ch, kernel, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(ch, ch, kernel, dilation=d, padding=(kernel-1)*d//2))
            for d in dilations])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(ch, ch, kernel, dilation=1, padding=(kernel-1)//2))
            for _ in dilations])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            r = x
            x = F.leaky_relu(x, 0.1)
            x = c1(x)
            x = F.leaky_relu(x, 0.1)
            x = c2(x)
            x = x + r
        return x

    def remove_weight_norm(self):
        for c in self.convs1: nn.utils.remove_weight_norm(c)
        for c in self.convs2: nn.utils.remove_weight_norm(c)


class HiFiGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        ch = VC["gen_init_ch"]
        self.pre = nn.utils.weight_norm(
            nn.Conv1d(N_MELS, ch, 7, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (r, k) in enumerate(zip(VC["upsample_rates"], VC["upsample_kernels"])):
            out_ch = ch // (2 ** (i + 1))
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(ch // (2**i), out_ch, k, stride=r,
                                   padding=(k-r)//2)))
            for kr, dr in zip(VC["resblock_kernels"], VC["resblock_dilations"]):
                self.resblocks.append(ResBlock(out_ch, kr, dr))

        final_ch = ch // (2 ** len(VC["upsample_rates"]))
        self.post = nn.utils.weight_norm(
            nn.Conv1d(final_ch, 1, 7, padding=3))
        self.n_res = len(VC["resblock_kernels"])

    def forward(self, mel):
        # mel: [B, T_mel, 80]  →  transpose to  [B, 80, T_mel]
        x = self.pre(mel.transpose(1, 2))
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            res = None
            for j in range(self.n_res):
                r = self.resblocks[i * self.n_res + j](x)
                res = r if res is None else res + r
            x = res / self.n_res
        x = F.leaky_relu(x, 0.1)
        x = torch.tanh(self.post(x))
        return x.squeeze(1)   # [B, T_wav]

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.pre)
        for up in self.ups: nn.utils.remove_weight_norm(up)
        for rb in self.resblocks: rb.remove_weight_norm()
        nn.utils.remove_weight_norm(self.post)


# ── Multi-Period Discriminator ────────────────────────────────────────────────
class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1,    32,  (5,1), (3,1), (2,0))),
            nn.utils.weight_norm(nn.Conv2d(32,   128, (5,1), (3,1), (2,0))),
            nn.utils.weight_norm(nn.Conv2d(128,  512, (5,1), (3,1), (2,0))),
            nn.utils.weight_norm(nn.Conv2d(512,  1024,(5,1), (3,1), (2,0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024,(5,1), 1,     (2,0))),
        ])
        self.post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3,1), 1, (1,0)))

    def forward(self, x):
        feats = []
        B, T = x.shape
        # Pad to be divisible by period
        pad = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad))
        x = x.view(B, 1, -1, self.period)
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1, -1), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList(
            [PeriodDiscriminator(p) for p in VC["mpd_periods"]])

    def forward(self, real, fake):
        r_outs, f_outs, r_feats, f_feats = [], [], [], []
        for d in self.discs:
            r, rf = d(real); f, ff = d(fake)
            r_outs.append(r); f_outs.append(f)
            r_feats.append(rf); f_feats.append(ff)
        return r_outs, f_outs, r_feats, f_feats


# ── Multi-Scale Discriminator ─────────────────────────────────────────────────
class ScaleDiscriminator(nn.Module):
    def __init__(self, norm=nn.utils.spectral_norm):
        super().__init__()
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    128,  15, 1,  7)),
            norm(nn.Conv1d(128,  128,  41, 2,  20, groups=4)),
            norm(nn.Conv1d(128,  256,  41, 2,  20, groups=16)),
            norm(nn.Conv1d(256,  512,  41, 4,  20, groups=16)),
            norm(nn.Conv1d(512,  1024, 41, 4,  20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, 1,  20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 5,  1,  2)),
        ])
        self.post = norm(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x):
        feats = []
        x = x.unsqueeze(1)
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1, -1), feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([
            ScaleDiscriminator(nn.utils.spectral_norm),
            ScaleDiscriminator(nn.utils.weight_norm),
            ScaleDiscriminator(nn.utils.weight_norm),
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, 2),
            nn.AvgPool1d(4, 2, 2),
        ])

    def forward(self, real, fake):
        r_outs, f_outs, r_feats, f_feats = [], [], [], []
        for i, d in enumerate(self.discs):
            if i > 0:
                real = self.pools[i-1](real)
                fake = self.pools[i-1](fake)
            r, rf = d(real); f, ff = d(fake)
            r_outs.append(r); f_outs.append(f)
            r_feats.append(rf); f_feats.append(ff)
        return r_outs, f_outs, r_feats, f_feats


# ── Vocoder losses ────────────────────────────────────────────────────────────
def disc_loss(r_outs, f_outs):
    """Least-squares GAN discriminator loss."""
    loss = 0.0
    for r, f in zip(r_outs, f_outs):
        loss += ((r - 1.0)**2).mean() + (f**2).mean()
    return loss

def gen_adv_loss(f_outs):
    """Least-squares GAN generator adversarial loss."""
    return sum(((f - 1.0)**2).mean() for f in f_outs)

def feature_matching_loss(r_feats_list, f_feats_list):
    """L1 distance between discriminator intermediate features."""
    loss = 0.0
    for r_feats, f_feats in zip(r_feats_list, f_feats_list):
        for r, f in zip(r_feats, f_feats):
            loss += F.l1_loss(f, r.detach())
    return loss

def mel_loss(fake_wav, real_wav):
    """L1 loss in mel-spectrogram domain (computed on GPU via STFT)."""
    # Build mel filterbank on device
    dev = fake_wav.device
    fb  = torch.tensor(
        librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT,
                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX),
        dtype=torch.float32, device=dev)  # [80, n_fft//2+1]
    win = torch.hann_window(WIN_LENGTH, device=dev)

    def to_mel(wav):
        s = torch.stft(wav, N_FFT, HOP_LENGTH, WIN_LENGTH,
                       win, return_complex=True)
        mag = s.abs().clamp(min=1e-5)      # [B, F, T]
        m   = torch.matmul(fb, mag)         # [B, 80, T]
        return torch.log(m)

    return F.l1_loss(to_mel(fake_wav), to_mel(real_wav))


# ── Vocoder dataset ───────────────────────────────────────────────────────────
class VocoderDataset(Dataset):
    def __init__(self, records, seg_frames):
        self.records   = records
        self.seg_wav   = seg_frames * HOP_LENGTH   # wav samples per segment

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        wav = load_wav(self.records[idx]["wav_path"])
        mel = wav_to_log_mel(wav)             # [T_mel, 80]
        T_mel = mel.shape[0]
        T_wav = len(wav)

        seg = self.seg_wav
        if T_wav <= seg:
            # pad short clips
            wav = np.pad(wav, (0, seg - T_wav + HOP_LENGTH))
            mel_target = mel.shape[0]
            mel = np.pad(mel, ((0, seg // HOP_LENGTH - mel.shape[0] + 1), (0,0)))

        # Random crop aligned between wav and mel
        max_start = len(wav) - seg
        if max_start <= 0:
            start_wav = 0
        else:
            start_wav = random.randint(0, max_start)
        start_mel = start_wav // HOP_LENGTH
        mel_seg   = VC["segment_frames"]

        mel_crop = mel[start_mel : start_mel + mel_seg]
        wav_crop = wav[start_wav : start_wav + seg]

        # Pad if needed
        if mel_crop.shape[0] < mel_seg:
            mel_crop = np.pad(mel_crop, ((0, mel_seg - mel_crop.shape[0]), (0,0)))
        if len(wav_crop) < seg:
            wav_crop = np.pad(wav_crop, (0, seg - len(wav_crop)))

        return (torch.tensor(mel_crop, dtype=torch.float32),   # [mel_seg, 80]
                torch.tensor(wav_crop, dtype=torch.float32))   # [seg_wav]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 TRAINER
# ─────────────────────────────────────────────────────────────────────────────
def train_vocoder(args, device, use_amp):
    work  = Path(args.work_dir)
    ckpts = work / "vocoder" / "checkpoints"
    samps = work / "vocoder" / "samples"
    plots = work / "vocoder" / "plots"
    for d in [ckpts, samps, plots]: d.mkdir(parents=True, exist_ok=True)

    EPOCHS = args.epochs or VC["epochs"]
    BS     = args.batch_size or VC["batch_size"]

    print("=" * 60)
    print("STAGE 2 — HiFi-GAN VOCODER TRAINING")
    print(f"  Data  : {args.data_dir}")
    print(f"  Work  : {work}/vocoder")
    print(f"  Epochs: {EPOCHS}   Batch: {BS}   Device: {device}")
    print("=" * 60)

    print("\nLoading metadata...")
    trn, val = load_metadata(Path(args.data_dir))

    trn_loader = DataLoader(
        VocoderDataset(trn, VC["segment_frames"]),
        batch_size=BS, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        VocoderDataset(val, VC["segment_frames"]),
        batch_size=BS, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print(f"  Train batches: {len(trn_loader)}  Val batches: {len(val_loader)}")

    gen = HiFiGANGenerator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    print(f"  Generator params  : {sum(p.numel() for p in gen.parameters()):,}")
    print(f"  MPD+MSD params    : "
          f"{sum(p.numel() for p in mpd.parameters())+sum(p.numel() for p in msd.parameters()):,}")

    opt_g = torch.optim.AdamW(gen.parameters(),
                               lr=VC["lr_g"], betas=(0.8, 0.99),
                               weight_decay=VC["weight_decay"])
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()),
                               lr=VC["lr_d"], betas=(0.8, 0.99),
                               weight_decay=VC["weight_decay"])
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, VC["lr_decay"])
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, VC["lr_decay"])
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # ── Checkpoint helpers ─────────────────────────────────────────────────
    def save(path, epoch, best_val, hist):
        torch.save({"epoch": epoch,
                    "gen": gen.state_dict(), "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": best_val, "history": hist}, path)
        print(f"  ✓ Checkpoint saved → {Path(path).name}")

    def load(path):
        ck = torch.load(path, map_location='cpu')
        gen.load_state_dict(ck["gen"])
        mpd.load_state_dict(ck["mpd"])
        msd.load_state_dict(ck["msd"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        sched_g.load_state_dict(ck["sched_g"])
        sched_d.load_state_dict(ck["sched_d"])
        scaler.load_state_dict(ck["scaler"])
        return ck["epoch"], ck["best_val"], ck.get("history", {
            "gen_loss": [], "disc_loss": [], "mel_loss": [], "val_mel": []})

    # ── Auto-resume ────────────────────────────────────────────────────────
    start, best_val = 1, float("inf")
    hist  = {"gen_loss": [], "disc_loss": [], "mel_loss": [], "val_mel": []}
    no_imp = 0

    latest = ckpts / "latest.pt"
    best   = ckpts / "best.pt"

    if args.resume and latest.exists():
        print(f"\nResuming from: {latest}")
        start, best_val, hist = load(latest)
        print(f"  Epoch {start}  best_val={best_val:.4f}")
        start += 1
    else:
        print("\nStarting vocoder from scratch (epoch 1)")

    # ── Validation ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate():
        gen.eval(); tot, n = 0.0, 0
        for mel, real in val_loader:
            mel=mel.to(device); real=real.to(device)
            fake = gen(mel)
            T = min(fake.size(1), real.size(1))
            tot += mel_loss(fake[:, :T], real[:, :T]).item() * mel.size(0)
            n   += mel.size(0)
        return tot / max(n, 1)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nTraining epoch {start} → {EPOCHS}")
    for epoch in range(start, EPOCHS + 1):
        gen.train(); mpd.train(); msd.train()
        g_tot = d_tot = m_tot = 0.0; n = 0; t0 = time.time()

        for mel, real in trn_loader:
            mel  = mel.to(device,  non_blocking=True)
            real = real.to(device, non_blocking=True)
            T    = real.size(1)

            # ── Discriminator step ──────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                fake = gen(mel).detach()
                fake_t = fake[:, :T]
                mpd_r, mpd_f, _, _ = mpd(real, fake_t)
                msd_r, msd_f, _, _ = msd(real, fake_t)
                loss_d = disc_loss(mpd_r, mpd_f) + disc_loss(msd_r, msd_f)

            opt_d.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.unscale_(opt_d)
            nn.utils.clip_grad_norm_(
                list(mpd.parameters()) + list(msd.parameters()), VC["grad_clip"])
            scaler.step(opt_d); scaler.update()

            # ── Generator step ──────────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                fake   = gen(mel)
                fake_t = fake[:, :T]
                _, mpd_f, mpd_rf, mpd_ff = mpd(real, fake_t)
                _, msd_f, msd_rf, msd_ff = msd(real, fake_t)
                loss_adv = gen_adv_loss(mpd_f) + gen_adv_loss(msd_f)
                loss_fm  = (feature_matching_loss(mpd_rf, mpd_ff) +
                            feature_matching_loss(msd_rf, msd_ff))
                loss_mel = mel_loss(fake_t, real)
                loss_g   = (loss_adv
                            + VC["lambda_fm"]  * loss_fm
                            + VC["lambda_mel"] * loss_mel)

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            nn.utils.clip_grad_norm_(gen.parameters(), VC["grad_clip"])
            scaler.step(opt_g); scaler.update()

            g_tot += loss_g.item() * mel.size(0)
            d_tot += loss_d.item() * mel.size(0)
            m_tot += loss_mel.item() * mel.size(0)
            n     += mel.size(0)

        sched_g.step(); sched_d.step()
        g_tot /= max(n, 1); d_tot /= max(n, 1); m_tot /= max(n, 1)
        v_mel = validate()
        elapsed = time.time() - t0

        hist["gen_loss"].append(g_tot)
        hist["disc_loss"].append(d_tot)
        hist["mel_loss"].append(m_tot)
        hist["val_mel"].append(v_mel)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"G={g_tot:.3f} D={d_tot:.3f} Mel={m_tot:.4f} | "
              f"ValMel={v_mel:.4f} | "
              f"LR={opt_g.param_groups[0]['lr']:.1e} | {elapsed:.0f}s")

        save(latest, epoch, best_val, hist)

        if epoch % VC["ckpt_every"] == 0:
            save(ckpts / f"epoch_{epoch:04d}.pt", epoch, best_val, hist)

        if v_mel < best_val:
            best_val = v_mel; no_imp = 0
            save(best, epoch, best_val, hist)
            print(f"  ★ New best: {best_val:.4f}")
        else:
            no_imp += 1
            if no_imp >= VC["patience"]:
                print("  Early stopping."); break

        if epoch % VC["ckpt_every"] == 0 or epoch == 1:
            # Generate a demo sample using ground-truth mel from val set
            try:
                gen.eval()
                mel_np = wav_to_log_mel(load_wav(val[0]["wav_path"]))
                mel_t  = torch.tensor(mel_np, dtype=torch.float32,
                                      device=device).unsqueeze(0)
                with torch.no_grad():
                    wav_t = gen(mel_t)
                wav_np = wav_t[0].float().cpu().numpy()
                out = str(samps / f"epoch_{epoch:04d}.wav")
                wav_np = wav_np / (np.abs(wav_np).max() + 1e-8)
                wav_int16 = (wav_np * 32767).astype(np.int16)
                import scipy.io.wavfile as wavfile
                wavfile.write(out, SAMPLE_RATE, wav_int16)
                print(f"  Sample: {out}")
            except Exception as e:
                print(f"  Sample failed: {e}")

            # Loss curve
            if len(hist["gen_loss"]) > 1:
                ep = list(range(1, len(hist["gen_loss"]) + 1))
                fig, ax = plt.subplots(1, 2, figsize=(14, 4))
                ax[0].plot(ep, hist["gen_loss"],  label="Gen")
                ax[0].plot(ep, hist["disc_loss"], label="Disc")
                ax[0].set_title("GAN Losses"); ax[0].legend(); ax[0].grid(True)
                ax[1].plot(ep, hist["mel_loss"],  label="Train Mel")
                ax[1].plot(ep, hist["val_mel"],   label="Val Mel")
                ax[1].set_title("Mel Loss"); ax[1].legend(); ax[1].grid(True)
                plt.tight_layout()
                plt.savefig(str(work / "vocoder" / "training_curve.png"), dpi=100)
                plt.close()

    print("\nVocoder training complete.")
    print(f"  Best checkpoint: {best}")


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — FULL PIPELINE SYNTHESIS
# ═════════════════════════════════════════════════════════════════════════════
def run_synthesis(args, device, use_amp):
    work = Path(args.work_dir)
    ac_best  = work / "acoustic" / "checkpoints" / "best.pt"
    voc_best = work / "vocoder"  / "checkpoints" / "best.pt"

    print("=" * 60)
    print("SYNTHESIS — Full Pipeline")
    print(f"  Text: {args.text}")
    print("=" * 60)

    # ── Load acoustic model ───────────────────────────────────────────────
    if not ac_best.exists():
        raise FileNotFoundError(f"Acoustic checkpoint not found: {ac_best}")
    print(f"\nLoading acoustic model from {ac_best}")
    ac_model = AcousticModel().to(device)
    ck = torch.load(ac_best, map_location='cpu')
    ac_model.load_state_dict(ck["model"])
    ac_model.eval()

    mel_np, attn_np = acoustic_synthesize(ac_model, args.text, device, use_amp)
    print(f"  Mel shape: {mel_np.shape}")

    # ── Load HiFi-GAN vocoder (if available, else Griffin-Lim) ───────────
    if voc_best.exists():
        print(f"Loading vocoder from {voc_best}")
        gen = HiFiGANGenerator().to(device)
        vc_ck = torch.load(voc_best, map_location='cpu')
        gen.load_state_dict(vc_ck["gen"])
        gen.remove_weight_norm()
        gen.eval()
        mel_t   = torch.tensor(mel_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            wav_t = gen(mel_t)
        audio = wav_t[0].float().cpu().numpy()
        print("  Using HiFi-GAN vocoder")
    else:
        print("  Vocoder not found — using Griffin-Lim (train vocoder for better quality)")
        audio = log_mel_to_audio_griffinlim(mel_np, n_iter=400)

    # ── Save output ────────────────────────────────────────────────────────
    out_dir = work / "synthesis_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts  = int(time.time())
    wav_path  = out_dir / f"output_{ts}.wav"
    plot_path = out_dir / f"output_{ts}.png"

    sf.write(str(wav_path), audio, SAMPLE_RATE)
    print(f"\n  Audio saved : {wav_path}")
    print(f"  Duration    : {len(audio)/SAMPLE_RATE:.2f}s")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].imshow(attn_np, aspect="auto", origin="lower")
    ax[0].set_title("Attention Alignment")
    ax[0].set_xlabel("Encoder steps"); ax[0].set_ylabel("Decoder steps")
    ax[1].imshow(mel_np.T, aspect="auto", origin="lower")
    ax[1].set_title("Predicted Mel Spectrogram")
    ax[1].set_xlabel("Time"); ax[1].set_ylabel("Mel bins")
    plt.suptitle(f'"{args.text[:60]}..."' if len(args.text)>60 else f'"{args.text}"')
    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=100)
    plt.close()
    print(f"  Plot saved  : {plot_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    use_amp = not args.no_amp and device.type == "cuda"

    print(f"\nDevice: {device}  |  GPUs: {n_gpus}  |  AMP: {use_amp}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if args.stage == "acoustic":
        train_acoustic(args, device, use_amp)
    elif args.stage == "vocoder":
        train_vocoder(args, device, use_amp)
    elif args.stage == "synthesize":
        run_synthesis(args, device, use_amp)


if __name__ == "__main__":
    main()
