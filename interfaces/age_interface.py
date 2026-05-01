import os
import json
from typing import Dict, Optional, Union

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


DEFAULT_IMG_SIZE = 224
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AgeResNet(nn.Module):
    """
    Pretrained ResNet18-based age regression model.

    Output:
        single scalar age prediction
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AgePredictionInterface:
    """
    Integration-ready interface for age prediction.

    Main usage:
        interface = AgePredictionInterface("best_age_resnet18.pth")
        result = interface.predict("face.jpg")
        print(result["age"])

    Returned result format:
        {
            "age": 24.6,
            "age_group": "young_adult"
        }
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.img_size = img_size

        self.model = AgeResNet()
        self.checkpoint = self._load_checkpoint(checkpoint_path)
        self._load_model_state(self.model, self.checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> Dict:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            return checkpoint

        raise ValueError("Checkpoint must be a dict / state_dict.")

    @staticmethod
    def _load_model_state(model: nn.Module, checkpoint: Dict) -> None:
        state = checkpoint.get("model_state", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)

        if missing:
            print(f"[Warning] Missing keys while loading model: {len(missing)}")
        if unexpected:
            print(f"[Warning] Unexpected keys while loading model: {len(unexpected)}")

    @staticmethod
    def _age_to_group(age: float) -> str:
        if age < 13:
            return "child"
        if age < 20:
            return "teen"
        if age < 35:
            return "young_adult"
        if age < 60:
            return "adult"
        return "senior"

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # (1, C, H, W)
        return tensor

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, Union[float, str]]:
        x = self.preprocess_image(image_path).to(self.device)
        pred_age = self.model(x).squeeze().item()

        # keep output realistic
        pred_age = max(0.0, float(pred_age))

        return {
            "age": round(pred_age, 2),
            "age_group": self._age_to_group(pred_age)
        }

    def get_model_info(self) -> Dict[str, Union[str, int, None]]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "model_type": "AgeResNet",
            "img_size": self.img_size,
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "checkpoint_stage": self.checkpoint.get("stage"),
        }


def load_age_interface(
    checkpoint_path: str,
    device: Optional[str] = None,
    img_size: int = DEFAULT_IMG_SIZE,
) -> AgePredictionInterface:
    return AgePredictionInterface(
        checkpoint_path=checkpoint_path,
        device=device,
        img_size=img_size,
    )


def predict_age(
    image_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Dict[str, Union[float, str]]:
    interface = load_age_interface(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return interface.predict(image_path=image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Age prediction inference interface")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to face image")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    interface = load_age_interface(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    print("Model info:")
    print(json.dumps(interface.get_model_info(), indent=2, ensure_ascii=False))
    print()

    result = interface.predict(args.image)
    print("Prediction:")
    print(json.dumps(result, indent=2, ensure_ascii=False))