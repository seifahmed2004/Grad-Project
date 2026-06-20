import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="CLI for the 4-model graduation project integration")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sign")
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", default=str(ROOT/"models/asl_landmark_best_v3.pth"))
    p.add_argument("--sentence", action="store_true")

    p = sub.add_parser("tts")
    p.add_argument("--text", required=True)
    p.add_argument("--acoustic", default=str(ROOT/"models/best_tts_acoustic.zip"))
    p.add_argument("--vocoder", default=None)
    p.add_argument("--output", default=str(ROOT/"outputs/tts_cli.wav"))

    p = sub.add_parser("speech")
    p.add_argument("--audio", required=True)
    p.add_argument("--checkpoint", default=str(ROOT/"models/best_speech_model.pth.zip"))

    p = sub.add_parser("age-gender")
    p.add_argument("--image", required=True)
    p.add_argument("--age", default=str(ROOT/"models/best_age_efficientnet_b4_finetuned.pth.zip"))
    p.add_argument("--gender", default=str(ROOT/"models/best_gender_utkface.pth.zip"))
    p.add_argument("--face", default=str(ROOT/"models/yolov8n-face-lindevs.pt.zip"))

    args = parser.parse_args()

    if args.cmd == "sign":
        from interfaces.sign_language_interface import load_sign_language_interface
        model = load_sign_language_interface(args.checkpoint)
        res = model.predict_sentence(args.video) if args.sentence else model.predict(args.video)
        print(res)

    elif args.cmd == "tts":
        from interfaces.tts_interface import load_tts_interface
        model = load_tts_interface(args.acoustic, args.vocoder)
        print(model.synthesize(args.text, args.output))

    elif args.cmd == "speech":
        from interfaces.speech_interface import load_speech_interface
        model = load_speech_interface(args.checkpoint)
        print(model.predict(args.audio))

    elif args.cmd == "age-gender":
        from interfaces.age_gender_interface import load_age_gender_interface
        model = load_age_gender_interface(args.age, args.gender, args.face)
        print(model.predict(args.image))

if __name__ == "__main__":
    main()
