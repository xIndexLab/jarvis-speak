"""Register or refresh a built-in voice preset in the ONNX manifest.

The ONNX backend resolves ``--voice <name>`` by looking up ``builtin_voices``
in ``models/MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json``. That manifest
lives under ``models/`` and is not version-controlled, but each entry's
``prompt_audio_codes`` is simply the audio-tokenizer encoding of a reference
wav. This script recomputes those codes from the committed reference audio and
writes the entry back into the manifest, so ``infer_onnx.py --voice <name>``
works on a fresh checkout without needing ``--prompt-audio-path``.

Example (regenerate the shipped Jarvis voice from assets/audio/jarvis.wav):

    python register_builtin_voice.py
    # or explicitly:
    python register_builtin_voice.py \
        --voice Jarvis \
        --audio assets/audio/jarvis.wav \
        --display-name "Jarvis (1.3x Adam)" \
        --group Custom

Prerequisites: conda env ``moss-tts-nano``; ONNX assets under ``./models``
(auto-downloaded on first run; if huggingface.co is unreachable, set
``HF_ENDPOINT=https://hf-mirror.com``).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from onnx_tts_runtime import (
    DEFAULT_BROWSER_ONNX_MODEL_DIR,
    OnnxTtsRuntime,
    _find_manifest_path,
)

REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register or refresh a built-in voice in the ONNX manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voice", default="Jarvis", help="Preset voice name.")
    parser.add_argument(
        "--audio",
        default="assets/audio/jarvis.wav",
        help="Reference wav (relative paths resolve against the repo root).",
    )
    parser.add_argument(
        "--display-name",
        default="Jarvis (1.3x Adam)",
        help="Display label stored in the manifest.",
    )
    parser.add_argument(
        "--group", default="Custom", help="Voice group label stored in the manifest."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="browser_onnx model directory. Defaults to ./models (auto-download).",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--execution-provider",
        choices=("cpu", "cuda"),
        default="cpu",
        help="onnxruntime execution provider. cuda requires onnxruntime-gpu.",
    )
    return parser


def resolve_audio_path(audio: str) -> Path:
    p = Path(audio).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = build_parser().parse_args()

    audio_path = resolve_audio_path(args.audio)
    if not audio_path.is_file():
        raise SystemExit(f"reference audio not found: {audio_path}")

    # Loads the manifest and the codec encode session; reads from ./models
    # (auto-downloads on first run if missing).
    runtime = OnnxTtsRuntime(
        model_dir=args.model_dir,
        thread_count=args.cpu_threads,
        max_new_frames=375,
        do_sample=True,
        sample_mode="fixed",
        execution_provider=args.execution_provider,
    )

    # Encode the reference wav with the audio tokenizer -> prompt_audio_codes.
    prompt_audio_codes = runtime.encode_reference_audio(audio_path)
    codes_shape = np.array(prompt_audio_codes).shape

    manifest_path = _find_manifest_path(DEFAULT_BROWSER_ONNX_MODEL_DIR.resolve())
    if manifest_path is None:
        raise SystemExit("browser_poc_manifest.json not found under ./models")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    entry = {
        "voice": args.voice,
        "display_name": args.display_name,
        "group": args.group,
        "audio_file": audio_path.name,
        "prompt_audio_codes": prompt_audio_codes,
    }
    # Idempotent: drop any existing entry with the same voice name, then append.
    manifest["builtin_voices"] = [
        v for v in manifest.get("builtin_voices", []) if v.get("voice") != args.voice
    ]
    manifest["builtin_voices"].append(entry)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info(
        "registered voice=%s audio=%s prompt_audio_codes=%s -> %s (total voices=%d)",
        args.voice,
        audio_path,
        codes_shape,
        manifest_path,
        len(manifest["builtin_voices"]),
    )


if __name__ == "__main__":
    main()
