from __future__ import annotations

import os
import sys
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict

import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UPLOADS_DIR = ROOT / "uploads" / "live_sign"
OUTPUTS_DIR = ROOT / "outputs"
SAMPLES_DIR = ROOT / "samples"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Ishara Sign Live API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

SIGN_API_WARMED_UP = False
SIGN_API_WARMUP_ERROR: Optional[str] = None
TTS_API_WARMED_UP = False
TTS_API_WARMUP_ERROR: Optional[str] = None
TTS_OUTPUT_CACHE: Dict[str, Path] = {}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _json_safe(obj: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if torch is not None and hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    return obj


def _error_response(exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        },
    )


def _save_upload(file: UploadFile, subdir: str = "chunks") -> Path:
    ext = Path(file.filename or "input.webm").suffix or ".webm"
    out_dir = UPLOADS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_timestamp()}{ext}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


def _normalize_video_for_model(input_path: Path, target_fps: int = 30, max_width: int = 640) -> Path:
    """
    Browser webm chunks may report bad FPS to OpenCV.
    Re-write to stable mp4 before the sign model sees it.
    """
    input_path = Path(input_path)
    normalized_dir = UPLOADS_DIR / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    output_path = normalized_dir / f"{input_path.stem}_normalized.mp4"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return input_path

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue

        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (max_width, int(h * scale)))
        frames.append(frame)

    cap.release()

    if not frames:
        return input_path

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(target_fps), (w, h))

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    return output_path


def _output_url(path: str | Path | None) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    try:
        return "/outputs/" + p.relative_to(OUTPUTS_DIR.resolve()).as_posix()
    except Exception:
        if p.exists():
            dst = OUTPUTS_DIR / p.name
            if p.resolve() != dst.resolve():
                shutil.copy2(p, dst)
            return "/outputs/" + dst.name
    return None


def _quick_hand_visible_check(video_path: Path, max_checked_frames: int = 45, sample_every: int = 2) -> Dict[str, Any]:
    """
    Optional release-to-confirm helper.
    Returns hand_visible=False only when no left/right hand appears in sampled frames.
    """
    try:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
    except Exception as exc:
        raise ImportError("MediaPipe legacy solutions are required for hand visibility check.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"hand_visible": False, "checked_frames": 0, "hand_frames": 0}

    checked_frames = 0
    hand_frames = 0
    frame_index = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25,
    ) as holistic:
        while checked_frames < max_checked_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index % sample_every != 0:
                continue

            checked_frames += 1
            h, w = frame.shape[:2]
            if w > 480:
                scale = 480 / float(w)
                frame = cv2.resize(frame, (480, int(h * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            if result.left_hand_landmarks or result.right_hand_landmarks:
                hand_frames += 1
                cap.release()
                return {"hand_visible": True, "checked_frames": checked_frames, "hand_frames": hand_frames}

    cap.release()
    return {"hand_visible": False, "checked_frames": checked_frames, "hand_frames": hand_frames}


def warmup_sign_model_once() -> None:
    """Preload the updated Sign model and optionally warm MediaPipe with samples/warmup_sign.mp4."""
    global SIGN_API_WARMED_UP, SIGN_API_WARMUP_ERROR

    if SIGN_API_WARMED_UP:
        return

    try:
        from model_adapters.sign_language import load_updated_sign_model, predict_single_sign

        load_updated_sign_model()

        warmup_video = SAMPLES_DIR / "warmup_sign.mp4"
        if warmup_video.exists():
            predict_single_sign(video_path=warmup_video, top_k=1)
            print("[SIGN LIVE API] Full warm-up done using samples/warmup_sign.mp4")
        else:
            print("[SIGN LIVE API] Model loaded. No warmup video found at samples/warmup_sign.mp4")

        SIGN_API_WARMED_UP = True
        SIGN_API_WARMUP_ERROR = None

    except Exception as exc:
        SIGN_API_WARMUP_ERROR = str(exc)
        print("[SIGN LIVE API] Warm-up failed:")
        print(traceback.format_exc())


def warmup_tts_once() -> None:
    """Preload Kokoro TTS and generate one tiny audio file before the first Speak Sentence click."""
    global TTS_API_WARMED_UP, TTS_API_WARMUP_ERROR

    if TTS_API_WARMED_UP:
        return

    try:
        from tts_inference import text_to_speech as kokoro_text_to_speech
        out_path = OUTPUTS_DIR / "_warmup_live_tts.wav"

        try:
            from tts_inference import get_pipeline
            get_pipeline("a")
        except Exception:
            pass

        try:
            kokoro_text_to_speech("Hello", output_path=str(out_path), gender="Male", age=34)
        except TypeError:
            try:
                kokoro_text_to_speech("Hello", str(out_path), gender="Male")
            except TypeError:
                kokoro_text_to_speech("Hello", str(out_path))

        TTS_API_WARMED_UP = True
        TTS_API_WARMUP_ERROR = None
        print("[SIGN LIVE API] TTS warm-up done.")

    except Exception as exc:
        TTS_API_WARMUP_ERROR = str(exc)
        print("[SIGN LIVE API] TTS warm-up failed:")
        print(traceback.format_exc())


@app.on_event("startup")
def startup_warmup():
    warmup_sign_model_once()
    warmup_tts_once()


@app.get("/api/sign-v2/warmup-status")
def warmup_status():
    return {
        "ok": SIGN_API_WARMED_UP and TTS_API_WARMED_UP,
        "sign_warmed_up": SIGN_API_WARMED_UP,
        "sign_error": SIGN_API_WARMUP_ERROR,
        "tts_warmed_up": TTS_API_WARMED_UP,
        "tts_error": TTS_API_WARMUP_ERROR,
    }



@app.get("/api/tts/warmup")
def tts_warmup():
    warmup_tts_once()
    return {
        "ok": TTS_API_WARMED_UP,
        "tts_warmed_up": TTS_API_WARMED_UP,
        "tts_error": TTS_API_WARMUP_ERROR,
    }


@app.post("/api/sign-v2/single")
def sign_single(
    video: UploadFile = File(...),
    top_k: int = Form(5),
    check_hand: bool = Form(False),
):
    try:
        raw_video_path = _save_upload(video, "single")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        if check_hand:
            hand_check = _quick_hand_visible_check(video_path)
            if not hand_check.get("hand_visible"):
                return {
                    "ok": True,
                    "result": {
                        "no_hand": True,
                        "hand_visible": False,
                        "message": "No hand detected in chunk.",
                        "hand_check": hand_check,
                        "top_k": [],
                    },
                }

        from model_adapters.sign_language import predict_single_sign
        result = predict_single_sign(video_path=video_path, top_k=int(top_k))
        result["raw_uploaded_video_path"] = str(raw_video_path)
        result["normalized_video_path"] = str(video_path)
        return {"ok": bool(result.get("ok", True)), "result": _json_safe(result)}

    except Exception as exc:
        return _error_response(exc)


@app.post("/api/sign-v2/sentence")
def sign_sentence(
    video: UploadFile = File(...),
    top_k: int = Form(5),
    threshold: float = Form(0.08),
    min_pause_sec: float = Form(0.35),
    min_segment_sec: float = Form(0.30),
    confidence_threshold: float = Form(0.08),
    use_language_decoder: bool = Form(True),
):
    try:
        raw_video_path = _save_upload(video, "sentence")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        from model_adapters.sign_language import predict_sign_sentence
        result = predict_sign_sentence(
            video_path=video_path,
            top_k=int(top_k),
            threshold=float(threshold),
            min_pause_sec=float(min_pause_sec),
            min_segment_sec=float(min_segment_sec),
            confidence_threshold=float(confidence_threshold),
            use_language_decoder=bool(use_language_decoder),
        )
        result["raw_uploaded_video_path"] = str(raw_video_path)
        result["normalized_video_path"] = str(video_path)
        return {"ok": bool(result.get("ok", True)), "result": _json_safe(result)}

    except Exception as exc:
        return _error_response(exc)


@app.post("/api/tts")
def text_to_speech(
    text: str = Form(...),
    gender: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    speed: Optional[str] = Form(None),
):
    try:
        from tts_inference import (
            text_to_speech as kokoro_text_to_speech,
            get_age_group,
            get_speed_for_age,
            get_voice_for_gender_and_age,
        )

        clean_text = " ".join(str(text or "").split())
        if not clean_text:
            raise ValueError("Input text is empty.")

        selected_speed = float(speed) if speed not in (None, "") else get_speed_for_age(age)
        selected_voice = voice if voice else get_voice_for_gender_and_age(gender, age)
        age_group = get_age_group(age)

        cache_key = f"{clean_text}|{gender}|{age}|{selected_voice}|{selected_speed}"
        cached_path = TTS_OUTPUT_CACHE.get(cache_key)
        if cached_path and Path(cached_path).exists():
            return {
                "ok": True,
                "result": {
                    "text": clean_text,
                    "output_path": str(cached_path),
                    "audio_url": _output_url(cached_path),
                    "gender": gender,
                    "age": age,
                    "age_group": age_group,
                    "voice": selected_voice,
                    "speed": selected_speed,
                    "cached": True,
                },
            }

        out_path = OUTPUTS_DIR / f"live_tts_{_timestamp()}.wav"
        kokoro_text_to_speech(
            text=clean_text,
            output_path=str(out_path),
            gender=gender,
            age=age,
            voice=selected_voice,
            speed=selected_speed,
        )

        TTS_OUTPUT_CACHE[cache_key] = out_path

        return {
            "ok": True,
            "result": {
                "text": clean_text,
                "output_path": str(out_path),
                "audio_url": _output_url(out_path),
                "gender": gender,
                "age": age,
                "age_group": age_group,
                "voice": selected_voice,
                "speed": selected_speed,
            },
        }

    except Exception as exc:
        return _error_response(exc)



# --- FAST STT API PATCH ---
import hashlib as _hashlib
import wave as _wave
import threading as _threading
from typing import Optional as _Optional, Any as _Any, Dict as _Dict

from fastapi import UploadFile as _UploadFile, File as _File

STT_API_CACHE: _Dict[str, _Dict[str, _Any]] = {}
STT_API_WARMED_UP = False
STT_API_WARMUP_ERROR: _Optional[str] = None


def _stt_sha(data: bytes) -> str:
    return _hashlib.sha256(data).hexdigest()


def _stt_extract_text(raw: _Any) -> str:
    if isinstance(raw, str):
        return raw.strip()

    if isinstance(raw, dict):
        return str(
            raw.get("text")
            or raw.get("transcript")
            or raw.get("clean_text")
            or raw.get("prediction")
            or raw.get("sentence")
            or ""
        ).strip()

    return str(raw or "").strip()


def _run_stt_adapter(audio_path: Path) -> _Dict[str, _Any]:
    import model_adapters.speech_to_text as speech_adapter

    candidate_names = [
        "transcribe_speech",
        "speech_to_text",
        "transcribe_audio",
        "predict_speech",
        "run_speech_to_text",
        "predict",
    ]

    last_error = None
    for name in candidate_names:
        fn = getattr(speech_adapter, name, None)
        if not callable(fn):
            continue

        attempts = [
            lambda: fn(audio_path=str(audio_path)),
            lambda: fn(audio_file=str(audio_path)),
            lambda: fn(file_path=str(audio_path)),
            lambda: fn(path=str(audio_path)),
            lambda: fn(str(audio_path)),
        ]

        for attempt in attempts:
            try:
                raw = attempt()
                return {
                    "ok": True,
                    "text": _stt_extract_text(raw),
                    "raw_result": _json_safe(raw) if "_json_safe" in globals() else raw,
                    "adapter_function": name,
                }
            except TypeError as exc:
                last_error = exc
                continue

    return {
        "ok": False,
        "error": "No compatible speech-to-text adapter function found in model_adapters/speech_to_text.py",
        "last_error": str(last_error) if last_error else None,
    }


def _make_silent_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    seconds = 1
    samples = b"\x00\x00" * sample_rate * seconds

    with _wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples)


def _stt_warmup_blocking() -> _Dict[str, _Any]:
    global STT_API_WARMED_UP, STT_API_WARMUP_ERROR

    if STT_API_WARMED_UP:
        return {
            "ok": True,
            "stt_warmed_up": True,
            "stt_error": STT_API_WARMUP_ERROR,
            "cached": True,
        }

    try:
        warm_path = OUTPUTS_DIR / "stt_warmup_silence.wav"
        _make_silent_wav(warm_path)

        # This may return empty text; the important part is loading the model once.
        _run_stt_adapter(warm_path)

        STT_API_WARMED_UP = True
        STT_API_WARMUP_ERROR = None
        return {"ok": True, "stt_warmed_up": True, "message": "STT warmup completed."}

    except Exception as exc:
        STT_API_WARMED_UP = False
        STT_API_WARMUP_ERROR = str(exc)
        return {"ok": False, "stt_warmed_up": False, "error": str(exc)}


def _stt_warmup_background() -> None:
    _thread = _threading.Thread(target=_stt_warmup_blocking, name="stt-warmup", daemon=True)
    _thread.start()


@app.on_event("startup")
def _startup_fast_stt_warmup():
    _stt_warmup_background()


@app.get("/api/stt/warmup")
def stt_warmup():
    return _stt_warmup_blocking()


@app.get("/api/stt/warmup-status")
def stt_warmup_status():
    return {
        "ok": STT_API_WARMED_UP,
        "stt_warmed_up": STT_API_WARMED_UP,
        "stt_error": STT_API_WARMUP_ERROR,
        "cache_size": len(STT_API_CACHE),
    }


@app.post("/api/stt")
async def speech_to_text_api(file: _UploadFile = _File(...)):
    try:
        data = await file.read()
        if not data:
            raise ValueError("Uploaded audio is empty.")

        digest = _stt_sha(data)
        if digest in STT_API_CACHE:
            cached = dict(STT_API_CACHE[digest])
            cached["cached"] = True
            return {"ok": True, "result": cached}

        ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
        if ext not in [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]:
            ext = ".wav"

        in_dir = OUTPUTS_DIR / "stt_api_inputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        audio_path = in_dir / f"stt_{digest[:24]}{ext}"
        if not audio_path.exists():
            audio_path.write_bytes(data)

        raw = _run_stt_adapter(audio_path)

        if not raw.get("ok"):
            return {"ok": False, "error": raw.get("error", "STT failed."), "result": raw}

        result = {
            "text": raw.get("text", ""),
            "audio_path": str(audio_path),
            "adapter_function": raw.get("adapter_function"),
            "raw": raw,
            "cached": False,
        }

        STT_API_CACHE[digest] = result
        return {"ok": True, "result": result}

    except Exception as exc:
        return _error_response(exc)
# --- END FAST STT API PATCH ---


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sign_live_api:app", host="127.0.0.1", port=8000, reload=False)
