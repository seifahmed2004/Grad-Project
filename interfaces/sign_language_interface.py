import os
import json
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


DEFAULT_IMG_SIZE = 224
DEFAULT_NUM_FRAMES = 8
DEFAULT_DROPOUT = 0.60
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TemporalResNet(nn.Module):
    """
    RGB-frame sign recognition model.

    Expected input shape:
        (B, T, C, H, W)
    where:
        B = batch size
        T = number of sampled frames
        C = 3
        H, W = image size
    """

    def __init__(self, num_classes: int, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        # Use weights=None to avoid download/environment issues during integration.
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)      # (B*T, F)
        feats = feats.view(b, t, -1)  # (B, T, F)
        feats = feats.mean(dim=1)     # temporal average
        feats = self.dropout(feats)
        out = self.head(feats)
        return out


class SignLanguageToTextInterface:
    """
    Integration-ready interface for the ASL RGB sign-language model.

    Main usage:
        interface = SignLanguageToTextInterface("asl_rgb_200_last.pth")
        result = interface.predict("sample_video.mp4")
        print(result["text"])

    Returned result format:
        {
            "text": "PREDICTED_GLOSS",
            "label_id": 12,
            "confidence": 0.83,
            "top_k": [
                {"label_id": 12, "text": "PREDICTED_GLOSS", "confidence": 0.83},
                ...
            ]
        }
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        img_size: int = DEFAULT_IMG_SIZE,
        num_frames: int = DEFAULT_NUM_FRAMES,
        dropout: float = DEFAULT_DROPOUT,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.img_size = img_size
        self.num_frames = num_frames
        self.dropout = dropout

        self.checkpoint: Dict = self._load_checkpoint(checkpoint_path)
        self.id2label: Dict[int, str] = self._extract_id2label(self.checkpoint)
        self.label2id: Dict[str, int] = {v: k for k, v in self.id2label.items()}
        self.num_classes = len(self.id2label)

        self.model = TemporalResNet(num_classes=self.num_classes, dropout=self.dropout)
        self._load_model_state(self.model, self.checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> Dict:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Checkpoint must be a dict. "
                "Use the full training checkpoint if possible."
            )
        return checkpoint

    @staticmethod
    def _extract_id2label(checkpoint: Dict) -> Dict[int, str]:
        if "id2label" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'id2label'. "
                "Use the full LAST checkpoint, not only the raw BEST state_dict."
            )

        raw = checkpoint["id2label"]
        return {int(k): str(v) for k, v in raw.items()}

    @staticmethod
    def _load_model_state(model: nn.Module, checkpoint: Dict) -> None:
        state = checkpoint.get("model_state", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)

        if missing:
            print(f"[Warning] Missing keys while loading model: {len(missing)}")
        if unexpected:
            print(f"[Warning] Unexpected keys while loading model: {len(unexpected)}")

    def _uniform_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return [0] * self.num_frames
        idxs = np.linspace(0, max(total_frames - 1, 0), self.num_frames).astype(int)
        return idxs.tolist()

    def _preprocess_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame_rgb, (self.img_size, self.img_size))
        frame = frame.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame = (frame - mean) / std
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        return frame

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No readable frames found in video: {video_path}")

        idxs = self._uniform_indices(len(frames))
        sampled = [frames[i] for i in idxs]
        sampled = [self._preprocess_frame(f) for f in sampled]

        x = np.stack(sampled, axis=0)  # (T, C, H, W)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, T, C, H, W)
        return x

    @torch.no_grad()
    def predict(
        self,
        video_path: str,
        top_k: int = 5
    ) -> Dict[str, Union[str, int, float, List[Dict[str, Union[int, str, float]]]]]:
        x = self.preprocess_video(video_path).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        k = min(top_k, probs.shape[1])
        top_probs, top_ids = torch.topk(probs, k=k, dim=1)

        top_probs = top_probs[0].detach().cpu().numpy().tolist()
        top_ids = top_ids[0].detach().cpu().numpy().tolist()

        top_results = []
        for label_id, conf in zip(top_ids, top_probs):
            top_results.append({
                "label_id": int(label_id),
                "text": self.id2label[int(label_id)],
                "confidence": float(conf),
            })

        best = top_results[0]
        return {
            "text": best["text"],
            "label_id": best["label_id"],
            "confidence": best["confidence"],
            "top_k": top_results,
        }

    def get_model_info(self) -> Dict[str, Union[str, int, float, List[str], Dict]]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "model_type": "TemporalResNet",
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "num_frames": self.num_frames,
            "dropout": self.dropout,
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "checkpoint_stage": self.checkpoint.get("stage"),
            "sample_labels": [self.id2label[i] for i in sorted(self.id2label.keys())[:10]],
        }


def load_sign_language_interface(
    checkpoint_path: str,
    device: Optional[str] = None,
    img_size: int = DEFAULT_IMG_SIZE,
    num_frames: int = DEFAULT_NUM_FRAMES,
    dropout: float = DEFAULT_DROPOUT,
) -> SignLanguageToTextInterface:
    return SignLanguageToTextInterface(
        checkpoint_path=checkpoint_path,
        device=device,
        img_size=img_size,
        num_frames=num_frames,
        dropout=dropout,
    )


def sign_to_text(
    video_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Union[str, int, float, List[Dict[str, Union[int, str, float]]]]]:
    interface = load_sign_language_interface(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return interface.predict(video_path=video_path, top_k=top_k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sign Language to Text inference interface"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to full .pth checkpoint"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-K predictions to return"
    )
    args = parser.parse_args()

    interface = load_sign_language_interface(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    print("Model info:")
    print(json.dumps(interface.get_model_info(), indent=2, ensure_ascii=False))
    print()

    result = interface.predict(args.video, top_k=args.top_k)
    print("Prediction:")
    print(json.dumps(result, indent=2, ensure_ascii=False))