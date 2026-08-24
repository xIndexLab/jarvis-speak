"""Command-line REPL test client for the Agent streaming TTS WebSocket endpoint.

Connects to ``ws://<host>:<port>/v1/tts/stream`` (the Agent incremental
streaming interface defined in app.py), drives the full上行 protocol
(start / text / flush / stop / cancel / ping), prints every下行 frame
(JSON control frames + binary audio frames with the 16-byte header), and
saves the received PCM into a .wav so you can verify the audio.

Two prompt sources for the ``start`` frame (mutually exclusive):
  --demo-id demo-1        use a built-in demo entry from assets/demo.jsonl
  --prompt-audio a.wav    load a local wav, base64-encode into prompt_audio_b64
                          (this is how you test custom / cloned voices such as
                          the Jarvis preset, which is NOT in demo.jsonl)

Usage examples:

  # Interactive REPL against a local ONNX server, using the Jarvis voice:
  python ws_tts_repl.py --prompt-audio assets/audio/jarvis.wav

  # Scripted / non-interactive (pipe commands on stdin); auto-stops on EOF:
  printf 'Hello, I am Jarvis.\\nAll systems are online.\\n/stop\\n' | \\
    python ws_tts_repl.py --prompt-audio assets/audio/jarvis.wav --auto-stop

  # One-shot: send a single text then stop, exit when audio done:
  python ws_tts_repl.py --demo-id demo-1 --text "你好，欢迎使用语音合成。" --auto-stop

REPL commands (one per line; non-interactive stdin also works):
  <any text>          send as a `text` delta frame
  /flush              force-flush the segmenter buffer
  /stop               send stop (finish) and keep receiving until `ended`
  /cancel             cancel the session
  /ping               send a ping (expects a pong)
  /send {raw json}    send an arbitrary JSON frame (advanced)
  /help               show commands
  /quit | /exit | EOF close the connection
  /status             print local counters
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import struct
import sys
import time
import wave
from pathlib import Path

import websockets

# WS audio frame header (mirror of app.py _build_ws_audio_frame)
_WS_AUDIO_MAGIC = 0xA1
_WS_AUDIO_VER = 0x01
_WS_FLAG_LAST = 0x01
_WS_FLAG_SILENCE = 0x02
_WS_FLAG_BOUNDARY = 0x04
_HEADER_STRUCT = struct.Struct(">BBBIIB")   # magic,ver,flags,seq,sr,channels (12B) + payload_len(I) (4B) = 16B

REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="REPL test client for the /v1/tts/stream WebSocket endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=18083)
    p.add_argument("--path", default="/v1/tts/stream", help="WS path on the server.")
    p.add_argument("--demo-id", default="", help="Built-in demo id, e.g. demo-1.")
    p.add_argument(
        "--prompt-audio",
        default="",
        help="Local wav path; base64-encoded into prompt_audio_b64 (takes priority over --demo-id).",
    )
    p.add_argument("--format", default="pcm", choices=("pcm",), help="Audio format negotiated in start frame.")
    p.add_argument("--first-segment-max-chars", type=int, default=24)
    p.add_argument("--segment-max-chars", type=int, default=60)
    p.add_argument("--segment-idle-timeout-ms", type=int, default=800)
    p.add_argument("--insert-inter-silence-ms", type=int, default=120)
    p.add_argument(
        "--strip-non-speech",
        choices=("1", "0"),
        default="1",
        help="Strip markdown / code blocks before synthesis.",
    )
    # ---- 生成参数覆盖（注入 start 帧 extra_tts_params，服务端 TtsSession 支持）----
    p.add_argument(
        "--do-sample",
        choices=("1", "0"),
        default="",
        help="Override do_sample. 0 forces greedy (deterministic EOS). Empty = use server default.",
    )
    p.add_argument(
        "--sample-mode",
        choices=("greedy", "fixed", "full"),
        default="",
        help="Override sample_mode. greedy=do_sample false. Empty = use server default.",
    )
    p.add_argument(
        "--max-new-frames",
        type=int,
        default=None,
        help="Override max_new_frames ceiling. None = use server default (375).",
    )
    p.add_argument(
        "--audio-temperature",
        type=float,
        default=None,
        help="Override audio_temperature. None = use server default.",
    )
    p.add_argument(
        "--audio-repetition-penalty",
        type=float,
        default=None,
        help="Override audio_repetition_penalty. None = use server default.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override sampling seed for reproducibility. None = use server default.",
    )
    p.add_argument(
        "--vad-silence-seconds",
        type=float,
        default=None,
        help="Override VAD trailing-silence cutoff (seconds). 0 disables VAD. None = use server default (0.6).",
    )
    p.add_argument(
        "--vad-silence-rms",
        type=float,
        default=None,
        help="Override VAD RMS threshold. None = use server default (0.01).",
    )
    p.add_argument(
        "--text",
        default="",
        help="One-shot: send this single text then /stop (implies --auto-stop).",
    )
    p.add_argument(
        "--auto-stop",
        action="store_true",
        help="Send /stop automatically when stdin reaches EOF (non-interactive mode).",
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "generated_audio"),
        help="Directory for the saved .pcm / .wav output.",
    )
    p.add_argument(
        "--name",
        default="",
        help="Output file basename. Defaults to ws_repl_<timestamp>.",
    )
    p.add_argument(
        "--quiet-bin",
        action="store_true",
        help="Do not print a line for every binary audio frame (still prints control frames).",
    )
    p.add_argument("--timeout", type=float, default=60.0, help="Max seconds to wait for `ended` after /stop.")
    return p


def resolve_start_payload(args: argparse.Namespace) -> dict:
    """Build the `start` JSON frame payload from CLI args."""
    payload: dict = {
        "type": "start",
        "format": args.format,
        "first_segment_max_chars": args.first_segment_max_chars,
        "segment_max_chars": args.segment_max_chars,
        "segment_idle_timeout_ms": args.segment_idle_timeout_ms,
        "insert_inter_silence_ms": args.insert_inter_silence_ms,
        "strip_non_speech": args.strip_non_speech,
    }
    if args.prompt_audio:
        wav_path = Path(args.prompt_audio).expanduser()
        if not wav_path.is_absolute():
            wav_path = (REPO_ROOT / wav_path).resolve()
        if not wav_path.is_file():
            raise SystemExit(f"prompt audio not found: {wav_path}")
        payload["prompt_audio_b64"] = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        payload["demo_id"] = ""
        logging.info("start: prompt_audio_b64 from %s (%d bytes raw)", wav_path, wav_path.stat().st_size)
    elif args.demo_id:
        payload["demo_id"] = args.demo_id
        logging.info("start: demo_id=%s", args.demo_id)
    else:
        raise SystemExit("provide --demo-id or --prompt-audio for the start frame")

    # 注入 extra_tts_params（仅含显式设置的项；服务端 TtsSession 会合并到 _DEFAULT_AGENT_TTS_PARAMS）
    extra: dict = {}
    if getattr(args, "do_sample", "") != "":
        extra["do_sample"] = bool(int(args.do_sample))
    if getattr(args, "sample_mode", "") != "":
        extra["sample_mode"] = args.sample_mode
    if args.max_new_frames is not None:
        extra["max_new_frames"] = int(args.max_new_frames)
    if args.audio_temperature is not None:
        extra["audio_temperature"] = float(args.audio_temperature)
    if args.audio_repetition_penalty is not None:
        extra["audio_repetition_penalty"] = float(args.audio_repetition_penalty)
    if args.seed is not None:
        extra["seed"] = int(args.seed)
    if args.vad_silence_seconds is not None:
        extra["vad_silence_seconds"] = float(args.vad_silence_seconds)
    if args.vad_silence_rms is not None:
        extra["vad_silence_rms"] = float(args.vad_silence_rms)
    if extra:
        payload["extra_tts_params"] = extra
        logging.info("start: extra_tts_params=%s", extra)
    return payload


def parse_bin_header(data: bytes) -> dict | None:
    """Parse the 16-byte WS audio frame header. Returns None if not a valid audio frame."""
    if len(data) < _HEADER_STRUCT.size + 4:  # 12-byte packed + 4-byte payload_len = 16
        return None
    magic, ver, flags, seq, sample_rate, channels = _HEADER_STRUCT.unpack_from(data, 0)
    (payload_len,) = struct.unpack_from(">I", data, _HEADER_STRUCT.size)
    if magic != _WS_AUDIO_MAGIC or ver != _WS_AUDIO_VER:
        return None
    payload = data[_HEADER_STRUCT.size + 4: _HEADER_STRUCT.size + 4 + payload_len]
    return {
        "magic": magic,
        "ver": ver,
        "flags": flags,
        "is_last": bool(flags & _WS_FLAG_LAST),
        "is_silence": bool(flags & _WS_FLAG_SILENCE),
        "is_boundary": bool(flags & _WS_FLAG_BOUNDARY),
        "seq": seq,
        "sample_rate": sample_rate,
        "channels": channels,
        "payload_len": payload_len,
        "payload": payload,
    }


class ReplClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.uri = f"ws://{args.host}:{args.port}{args.path}"
        stem = args.name or f"ws_repl_{int(time.time())}"
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.pcm_path = out_dir / f"{stem}.pcm"
        self.wav_path = out_dir / f"{stem}.wav"
        self.pcm_buf = bytearray()
        self.sample_rate = 0
        self.channels = 0
        self.bin_count = 0
        self.audio_bytes = 0
        self.silence_bytes = 0
        self.ended = False
        self.ended_reason: str | None = None
        self.last_error: dict | None = None
        self.start_time = time.monotonic()

    # ---- pretty printers ----
    @staticmethod
    def _fmt_flags(info: dict) -> str:
        tags = []
        if info["is_last"]:
            tags.append("LAST")
        if info["is_silence"]:
            tags.append("SILENCE")
        if info["is_boundary"]:
            tags.append("BOUNDARY")
        return ",".join(tags) or "-"

    def _print_bin(self, info: dict) -> None:
        if self.args.quiet_bin and not info["is_last"]:
            return
        dur_ms = (len(info["payload"]) / (info["sample_rate"] * info["channels"] * 2) * 1000.0) if info["sample_rate"] else 0.0
        print(
            f"[bin] seq={info['seq']:>4} len={info['payload_len']:>7} "
            f"sr={info['sample_rate']} ch={info['channels']} flags={self._fmt_flags(info)} "
            f"({dur_ms:.0f}ms)"
        )

    def _print_ctrl(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "ready":
            print(f"[ctrl] READY session={msg.get('session_id')} sr={msg.get('sample_rate')} ch={msg.get('channels')} codec={msg.get('codec')}")
        elif mtype == "sentence":
            print(f"[ctrl] SENTENCE idx={msg.get('index')} first={msg.get('is_first')} text={msg.get('text')!r}")
        elif mtype == "pong":
            print("[ctrl] PONG")
        elif mtype == "error":
            self.last_error = msg
            print(f"[ctrl] ERROR code={msg.get('code')} fatal={msg.get('fatal')} msg={msg.get('message')}")
        elif mtype == "ended":
            self.ended = True
            self.ended_reason = str(msg.get("reason"))
            print(f"[ctrl] ENDED reason={msg.get('reason')}")
        else:
            print(f"[ctrl] {msg}")

    # ---- receiver loop ----
    async def receive_loop(self, ws) -> None:
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    info = parse_bin_header(message)
                    if info is None:
                        print(f"[bin?] unparsed binary frame, {len(message)} bytes")
                        continue
                    self.bin_count += 1
                    if info["sample_rate"]:
                        self.sample_rate = info["sample_rate"]
                    if info["channels"]:
                        self.channels = info["channels"]
                    if info["payload"]:
                        if info["is_silence"]:
                            self.silence_bytes += len(info["payload"])
                        else:
                            self.pcm_buf.extend(info["payload"])
                            self.audio_bytes += len(info["payload"])
                    self._print_bin(info)
                elif isinstance(message, str):
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        print(f"[text?] {message}")
                        continue
                    self._print_ctrl(msg)
                    if msg.get("type") == "error" and msg.get("fatal"):
                        self.ended = True
                        self.ended_reason = "fatal_error"
        except websockets.ConnectionClosed:
            pass

    # ---- sender helpers ----
    async def send_json(self, ws, obj: dict) -> None:
        await ws.send(json.dumps(obj, ensure_ascii=False))
        logging.debug("sent %s", obj.get("type"))

    # ---- stdin reader (non-blocking via thread) ----
    async def stdin_loop(self, ws) -> None:
        loop = asyncio.get_running_loop()
        auto_stop = self.args.auto_stop or bool(self.args.text)
        eof_seen = False

        # one-shot text mode: send the text, then stop, then exit stdin loop
        if self.args.text:
            await self.send_json(ws, {"type": "text", "delta": self.args.text})
            await self.send_json(ws, {"type": "stop"})
            return

        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except (ValueError, OSError):
                # stdin closed (e.g. event loop shut down)
                break
            if line == "":
                # EOF
                if not eof_seen:
                    eof_seen = True
                    if auto_stop and not self.ended:
                        print("[repl] stdin EOF -> sending /stop")
                        try:
                            await self.send_json(ws, {"type": "stop"})
                        except websockets.ConnectionClosed:
                            break
                # keep the loop alive so receiver can finish; break on ended/closed
                if self.ended:
                    break
                await asyncio.sleep(0.05)
                continue
            line = line.rstrip("\n")
            if not line:
                continue
            cmd = line.strip()
            try:
                if cmd in ("/quit", "/exit"):
                    break
                elif cmd == "/help":
                    self.print_help()
                elif cmd == "/status":
                    self.print_status()
                elif cmd == "/flush":
                    await self.send_json(ws, {"type": "flush"})
                    print("[repl] sent flush")
                elif cmd == "/stop":
                    await self.send_json(ws, {"type": "stop"})
                    print("[repl] sent stop")
                elif cmd == "/cancel":
                    await self.send_json(ws, {"type": "cancel"})
                    print("[repl] sent cancel")
                elif cmd == "/ping":
                    await self.send_json(ws, {"type": "ping"})
                    print("[repl] sent ping")
                elif cmd.startswith("/send "):
                    raw = cmd[len("/send "):].strip()
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        print(f"[repl] /send needs valid JSON: {exc}")
                    else:
                        await self.send_json(ws, obj)
                        print(f"[repl] sent {obj.get('type')}")
                else:
                    # plain text -> a `text` delta frame
                    await self.send_json(ws, {"type": "text", "delta": line})
            except websockets.ConnectionClosed:
                break

    def print_help(self) -> None:
        print(
            "REPL commands:\n"
            "  <text>           send as a `text` delta frame\n"
            "  /flush           force-flush segmenter buffer\n"
            "  /stop            finish + keep receiving until `ended`\n"
            "  /cancel          cancel the session\n"
            "  /ping            send ping (expects pong)\n"
            "  /send {json}     send an arbitrary JSON frame\n"
            "  /status          print local counters\n"
            "  /quit | /exit    close\n"
        )

    def print_status(self) -> None:
        dur_s = len(self.pcm_buf) / (self.sample_rate * self.channels * 2) if self.sample_rate and self.channels else 0.0
        print(
            f"[status] state={'ended' if self.ended else 'open'} "
            f"bin_frames={self.bin_count} audio_bytes={self.audio_bytes} "
            f"silence_bytes={self.silence_bytes} sr={self.sample_rate} ch={self.channels} "
            f"pcm_seconds={dur_s:.2f}"
        )

    # ---- output finalization ----
    def finalize_output(self) -> None:
        if not self.pcm_buf:
            print("[out] no audio payload captured; skipping wav write")
            return
        # write raw pcm
        self.pcm_path.write_bytes(bytes(self.pcm_buf))
        # wrap into a wav
        sr = self.sample_rate or 24000
        ch = self.channels or 1
        with wave.open(str(self.wav_path), "wb") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(2)  # pcm s16le
            wf.setframerate(sr)
            wf.writeframes(bytes(self.pcm_buf))
        dur = len(self.pcm_buf) / (sr * ch * 2)
        print(f"[out] saved pcm={self.pcm_path} ({len(self.pcm_buf)} bytes)")
        print(f"[out] saved wav={self.wav_path} ({sr}Hz {ch}ch {dur:.2f}s)")

    async def run(self) -> int:
        start_payload = resolve_start_payload(self.args)
        print(f"[repl] connecting {self.uri}")
        try:
            async with websockets.connect(self.uri, max_size=None) as ws:
                await self.send_json(ws, start_payload)
                print("[repl] sent start; waiting for ready...")
                recv_task = asyncio.create_task(self.receive_loop(ws))
                stdin_task = asyncio.create_task(self.stdin_loop(ws))
                # wait until ended or stdin done, with a hard timeout
                try:
                    done, pending = await asyncio.wait(
                        {recv_task, stdin_task},
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=None if self.args.text else None,
                    )
                except asyncio.CancelledError:
                    pass
                # if stdin finished first, still wait for `ended` (bounded)
                if not self.ended and not stdin_task.done():
                    pass
                # drain: wait for ended within timeout
                if not self.ended:
                    try:
                        await asyncio.wait_for(self._wait_ended(), timeout=self.args.timeout)
                    except asyncio.TimeoutError:
                        print(f"[repl] timeout: no `ended` within {self.args.timeout}s; closing")
                # cancel stdin loop if still running
                stdin_task.cancel()
                try:
                    await ws.close()
                except Exception:
                    pass
                recv_task.cancel()
        except websockets.InvalidURI:
            print(f"[repl] invalid URI: {self.uri}")
            return 1
        except OSError as exc:
            print(f"[repl] connection failed: {exc}")
            print("[repl] is the server running? start it with: moss-tts-nano serve --backend onnx")
            return 1

        self.finalize_output()
        self.print_status()
        elapsed = time.monotonic() - self.start_time
        print(f"[repl] done in {elapsed:.1f}s; ended_reason={self.ended_reason}")
        return 0 if (self.ended and not (self.last_error and self.last_error.get("fatal"))) else 1

    async def _wait_ended(self) -> None:
        while not self.ended:
            await asyncio.sleep(0.05)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO if not args.quiet_bin else logging.WARNING,
    )
    return asyncio.run(ReplClient(args).run())


if __name__ == "__main__":
    raise SystemExit(main())
