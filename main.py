import os
import json
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

SIGN_CHECKPOINT = os.path.join(MODELS_DIR, "asl_rgb_200_last.pth")
TTS_CHECKPOINT = os.path.join(MODELS_DIR, "best.pt")
SPEECH_CHECKPOINT = os.path.join(MODELS_DIR, "best_speech_model.pth")
AGE_CHECKPOINT = os.path.join(MODELS_DIR, "best_age_resnet18.pth")

os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_sign_to_text(video_path, device=None, top_k=5):
    from interfaces.sign_language_interface import load_sign_language_interface

    sign_interface = load_sign_language_interface(
        checkpoint_path=SIGN_CHECKPOINT,
        device=device
    )

    result = sign_interface.predict(video_path=video_path, top_k=top_k)

    save_json(result, os.path.join(OUTPUTS_DIR, "sign_result.json"))
    return result


def run_text_to_speech(text, device=None, output_audio_path=None):
    from interfaces.tts_interface import load_tts_interface

    if output_audio_path is None:
        output_audio_path = os.path.join(OUTPUTS_DIR, "tts_output.wav")

    tts_interface = load_tts_interface(
        checkpoint_path=TTS_CHECKPOINT,
        device=device
    )

    result = tts_interface.synthesize(
        text=text,
        output_audio_path=output_audio_path
    )

    save_json(result, os.path.join(OUTPUTS_DIR, "tts_result.json"))
    return result


def run_sign_to_speech(video_path, device=None, top_k=5, output_audio_path=None):
    from interfaces.sign_language_interface import load_sign_language_interface
    from interfaces.tts_interface import load_tts_interface

    if output_audio_path is None:
        output_audio_path = os.path.join(OUTPUTS_DIR, "sign_tts_output.wav")

    sign_interface = load_sign_language_interface(
        checkpoint_path=SIGN_CHECKPOINT,
        device=device
    )
    tts_interface = load_tts_interface(
        checkpoint_path=TTS_CHECKPOINT,
        device=device
    )

    sign_result = sign_interface.predict(video_path=video_path, top_k=top_k)
    predicted_text = sign_result["text"]

    tts_result = tts_interface.synthesize(
        text=predicted_text,
        output_audio_path=output_audio_path
    )

    result = {
        "pipeline": "sign_language_to_text_to_speech",
        "input_video_path": video_path,
        "predicted_text": predicted_text,
        "sign_result": sign_result,
        "tts_result": tts_result,
    }

    save_json(result, os.path.join(OUTPUTS_DIR, "sign_tts_result.json"))
    return result


def run_speech_to_text(audio_path, device=None, use_beam=True):
    from interfaces.speech_interface import load_speech_interface

    speech_interface = load_speech_interface(
        checkpoint_path=SPEECH_CHECKPOINT,
        device=device
    )

    result = speech_interface.predict(audio_path=audio_path, use_beam=use_beam)

    save_json(result, os.path.join(OUTPUTS_DIR, "speech_result.json"))
    return result


def run_age_prediction(image_path, device=None):
    from interfaces.age_interface import load_age_interface

    age_interface = load_age_interface(
        checkpoint_path=AGE_CHECKPOINT,
        device=device
    )

    result = age_interface.predict(image_path=image_path)

    save_json(result, os.path.join(OUTPUTS_DIR, "age_result.json"))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-model integration main runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("sign-text", help="Run sign language to text")
    p1.add_argument("--video", type=str, required=True, help="Path to input video")
    p1.add_argument("--device", type=str, default=None, help="cpu or cuda")
    p1.add_argument("--top_k", type=int, default=5, help="Top-K predictions")

    p2 = subparsers.add_parser("text-tts", help="Run text to speech")
    p2.add_argument("--text", type=str, required=True, help="Input text")
    p2.add_argument("--device", type=str, default=None, help="cpu or cuda")
    p2.add_argument("--output", type=str, default=None, help="Output wav path")

    p3 = subparsers.add_parser("sign-tts", help="Run sign language to text to speech")
    p3.add_argument("--video", type=str, required=True, help="Path to input video")
    p3.add_argument("--device", type=str, default=None, help="cpu or cuda")
    p3.add_argument("--top_k", type=int, default=5, help="Top-K predictions")
    p3.add_argument("--output", type=str, default=None, help="Output wav path")

    p4 = subparsers.add_parser("speech-text", help="Run speech to text")
    p4.add_argument("--audio", type=str, required=True, help="Path to input audio")
    p4.add_argument("--device", type=str, default=None, help="cpu or cuda")
    p4.add_argument("--decoder", type=str, choices=["beam", "greedy"], default="beam")

    p5 = subparsers.add_parser("age", help="Run age prediction")
    p5.add_argument("--image", type=str, required=True, help="Path to input image")
    p5.add_argument("--device", type=str, default=None, help="cpu or cuda")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sign-text":
        result = run_sign_to_text(
            video_path=args.video,
            device=args.device,
            top_k=args.top_k,
        )

    elif args.command == "text-tts":
        result = run_text_to_speech(
            text=args.text,
            device=args.device,
            output_audio_path=args.output,
        )

    elif args.command == "sign-tts":
        result = run_sign_to_speech(
            video_path=args.video,
            device=args.device,
            top_k=args.top_k,
            output_audio_path=args.output,
        )

    elif args.command == "speech-text":
        result = run_speech_to_text(
            audio_path=args.audio,
            device=args.device,
            use_beam=(args.decoder == "beam"),
        )

    elif args.command == "age":
        result = run_age_prediction(
            image_path=args.image,
            device=args.device,
        )

    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()