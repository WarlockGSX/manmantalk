#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音转写：音频文件 -> 逐字稿（保留原始字段，不做加工）

两条路，脚本自己选：

  本地（默认）   faster-whisper，全程离线。不需要账号、密钥或付费。
                 缺点：不做说话人分离。
  腾讯云          设好密钥后自动启用。支持说话人分离（说话人编号）。
                 缺点：录音会上传到腾讯云，按量计费。

只要有 TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY 环境变量，
就自动走腾讯云；没有就走本地。加 --provider local 可以强制本地（不上传）。

用法：
    python transcribe.py <音频文件> [--out ./output] [--provider auto|local|tencent]
    python transcribe.py <音频文件> --codec mp3          # 编码格式（默认 opus）
    python transcribe.py <音频文件> --dump-raw           # 存下云端原始响应

腾讯云需要（只在走云端时要）：
    TENCENTCLOUD_SECRET_ID     腾讯云 SecretId
    TENCENTCLOUD_SECRET_KEY    腾讯云 SecretKey
    TENCENTCLOUD_REGION        地域，默认 ap-guangzhou

输出：
    <out>/<文件名>.jsonl              每行一段，字段见下
    <out>/<文件名>.md                 带时间戳的可读逐字稿
    <out>/<文件名>.raw.json           --dump-raw 时的云端原始响应

jsonl 每行字段：
    start / end          片段起止秒数（已换算为整段音频的绝对时间）
    text                 该段文字
    speaker              说话人编号。**只在同一 chunk 内可比**，跨 chunk 不可比；
                         且它只是声学聚类，不是角色（谁是谁要靠上层确认）
    chunk                所属切片序号（从 0 开始）。未切片时恒为 0

    —— 以下只在本地路径出现（Whisper 模型原始输出，不做换算）——
    avg_logprob          Whisper 的平均对数概率（负数，越接近 0 越有把握）
    no_speech_prob       该段「没有人声」的概率
    compression_ratio    文本重复度（明显偏高提示可能是重复幻觉）
    temperature          解码温度

    —— 以下两个路径都会有 ——
    rms_dbfs             该段音量（均方根，dBFS）——由音频算出，非模型输出
    peak_dbfs            该段峰值音量（dBFS）——由音频算出，非模型输出
    chars_per_sec        语速（字数/秒）——由文字与时长算出
    words                词级时间戳（本地路径需 --words；云端路径取决于返回）

注意：两个路径都只做转写。说话人编号是声学聚类，进入分析前必须由使用者确认角色。
"""

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------
# 国内可访问性：HuggingFace 直连在国内经常不通，模型会下载失败。
# 默认走镜像站。如果你能直连 HF，先设置 HF_ENDPOINT 环境变量即可覆盖。
# ---------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SUPPORTED = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".mp4", ".amr"}

# 判定「这段没听清」的阈值（作用于 avg_logprob 原值，不是加工后的概率）。
# 参考量级：-0.25 很稳，-1.0 一般，-1.5 以下基本不可信。
# 注意：这个阈值是按经验设的，应当用真实录音校准后再固化。
AVG_LOGPROB_FLOOR = -1.0

# ===============================================================
# 腾讯云录音文件识别
# ===============================================================
TENCENT_HOST = "asr.tencentcloudapi.com"
TENCENT_SERVICE = "asr"
TENCENT_VERSION = "2019-06-14"
# 必须与官方 SDK 逐字一致：官方签的就是这个值（不带 charset），
# 且参与签名的头只有 content-type 和 host 两个。
TENCENT_CONTENT_TYPE = "application/json"

# 接口限制：Data（base64 后的音频）必须 < 5MB。留一点余量。
TENCENT_HARD_LIMIT = 5 * 1024 * 1024
TENCENT_SAFE_LIMIT = int(4.6 * 1024 * 1024)

# 语言 -> 引擎。腾讯云 16k 系列只支持单声道，正好符合说话人分离的要求。
LANG_TO_ENGINE = {
    "zh": "16k_zh",
    "en": "16k_en",
    "yue": "16k_yue",
}

# 码率阶梯：从高到低试，尽量让整段音频塞进一次请求。
# 单次请求意味着说话人编号全局一致，这比多切一段更值钱。
BITRATE_LADDER = {
    "opus": [64, 48, 32, 24, 16],
    "mp3": [128, 96, 64, 48, 32],
}
# 低于这个码率就不再降了 —— 宁可切片，也不把音频压到影响识别。
BITRATE_FLOOR = {"opus": 24, "mp3": 32}


def die(msg: str) -> None:
    print(f"\n[错误] {msg}\n", file=sys.stderr)
    sys.exit(1)


def fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def check_deps_local() -> None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        die(
            "未安装 faster-whisper，本地转写不可用。请先执行：\n"
            "    pip install faster-whisper\n"
            "或设好腾讯云密钥改用云端转写。"
        )


# ---------------------------------------------------------------
# 音量：由音频本身算出的客观值，不是模型输出。
# 用 PyAV（它本来就是 faster-whisper 的依赖，不需要另外安装）。
# 任何一步失败就返回 None，音量字段留空——**不编数**。
# ---------------------------------------------------------------
def decode_mono(path, sr: int = 16000):
    try:
        import av
        import numpy as np
    except ImportError:
        return None, None
    try:
        container = av.open(str(path))
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=sr
        )
        chunks = []
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
        try:  # 冲出重采样器里剩下的一点尾巴
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray().reshape(-1))
        except Exception:
            pass
        container.close()
        if not chunks:
            return None, sr
        return np.concatenate(chunks).astype(np.float32), sr
    except Exception:
        return None, None


def dbfs(x, mode: str = "rms"):
    """把一段样本转成 dBFS。空段或全静音返回 None。"""
    if x is None or x.size == 0:
        return None
    try:
        import numpy as np

        if mode == "peak":
            v = float(np.max(np.abs(x)))
        else:
            v = float(np.sqrt(np.mean(np.square(x))))
        return round(20 * math.log10(v), 2) if v > 1e-9 else -120.0
    except Exception:
        return None


def slice_at(audio, sr, start: float, end: float):
    if audio is None or sr is None:
        return None
    i = max(0, int(start * sr))
    j = min(audio.size, int(end * sr))
    return audio[i:j] if j > i else None


# ---------------------------------------------------------------
# 编码：把 16k 单声道波形压成 mp3 / ogg-opus，塞进腾讯云的 5MB 限额。
# 只有本地解码得出音频时才可能——解码失败就退回「原样提交」。
# ---------------------------------------------------------------
def encode_audio(audio, sr: int, codec: str, bitrate_kbps: int, out_path: Path) -> bool:
    """把 float32 单声道波形编码成文件。成功返回 True。

    失败不抛异常：调用方需要据此改走别的策略，异常会让错误信息变难懂。
    """
    try:
        import av
    except ImportError:
        return False
    try:
        fmt = "mp3" if codec == "mp3" else "ogg"
        container = av.open(str(out_path), mode="w", format=fmt)
        stream = container.add_stream("libmp3lame" if codec == "mp3" else "libopus", rate=sr)
        stream.bit_rate = int(bitrate_kbps) * 1000
        frame = av.AudioFrame.from_ndarray(
            audio.reshape(1, -1).astype("float32"), format="fltp", layout="mono"
        )
        frame.sample_rate = sr
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def b64_len(raw_bytes: int) -> int:
    """base64 之后的长度（含换行前的理论值）。"""
    return 4 * ((raw_bytes + 2) // 3)


def plan_cloud(audio, sr: int, codec: str, probe_path: Path, limit: int):
    """**实测**决定码率与切片 —— 不用码率估算，因为编码器给不准。

    实测过：libopus 在低码率下实际输出只有标称码率的 0.67–0.74 倍，
    按标称码率估算会过于保守，把本来能一次传完的录音白白切碎，
    而切片会让说话人编号失去全局一致性。

    先按码率从高到低整段真编码一遍，取第一个真实体积塞得进限额的码率；
    都塞不下，就用下限码率，按实测字节率反推每片时长。

    返回 (mode, bitrate_kbps, chunk_seconds, probe_bytes)；失败返回 None。
    """
    duration = audio.size / sr
    ladder = [b for b in BITRATE_LADDER[codec] if b >= BITRATE_FLOOR[codec]]
    for br in ladder:
        if not encode_audio(audio, sr, codec, br, probe_path):
            return None
        size = probe_path.stat().st_size
        if b64_len(size) <= limit:
            return ("single", br, None, size)
    if not encode_audio(audio, sr, codec, BITRATE_FLOOR[codec], probe_path):
        return None
    size = probe_path.stat().st_size
    bytes_per_sec = size / duration
    # base64 后要 ≤ limit，反推原始字节上限；再留 3% 余量给容器尾部开销
    max_raw = int((limit - 2048) * 3 / 4)
    chunk_sec = max(60.0, max_raw / bytes_per_sec * 0.97)
    return ("chunked", BITRATE_FLOOR[codec], chunk_sec, size)


# ---------------------------------------------------------------
# 腾讯云 TC3-HMAC-SHA256 签名
# 纯标准库实现，不依赖 tencentcloud-sdk-python。
# 已用官方文档给出的完整测试向量验证（见仓库文档）。
# ---------------------------------------------------------------
def _tc3_signature(secret_key: str, date: str, service: str, string_to_sign: str) -> str:
    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    secret_date = _hmac(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = _hmac(secret_date, service)
    secret_signing = _hmac(secret_service, "tc3_request")
    return hmac.new(
        secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def build_canonical_request(
    method: str, uri: str, query: str, headers: dict, signed_headers: list, payload: bytes
) -> str:
    canon_headers = "".join(
        f"{h.lower()}:{headers[h].strip()}\n" for h in signed_headers
    )
    signed = ";".join(h.lower() for h in signed_headers)
    hashed_payload = hashlib.sha256(payload).hexdigest()
    return (
        method + "\n"
        + uri + "\n"
        + query + "\n"
        + canon_headers + "\n"
        + signed + "\n"
        + hashed_payload
    )


def tc3_authorization(
    secret_id: str,
    secret_key: str,
    service: str,
    host: str,
    action: str,
    payload: bytes,
    timestamp: int,
) -> str:
    """按腾讯云官方 Python SDK 的算法拼 Authorization。

    与官方实现逐字节对齐的两个要点（都对照 SDK 源码核过）：
      1. 参与签名的头**只有** content-type 和 host，不含 x-tc-action；
      2. content-type 取 "application/json"，不带 charset。
    action 参数保留只为可读性，不参与签名。
    """
    date = datetime.datetime.fromtimestamp(
        timestamp, datetime.timezone.utc
    ).strftime("%Y-%m-%d")
    headers = {"content-type": TENCENT_CONTENT_TYPE, "host": host}
    signed_headers = ["content-type", "host"]
    canonical = build_canonical_request("POST", "/", "", headers, signed_headers, payload)
    scope = f"{date}/{service}/tc3_request"
    string_to_sign = "TC3-HMAC-SHA256\n%d\n%s\n%s" % (
        timestamp,
        scope,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
    signature = _tc3_signature(secret_key, date, service, string_to_sign)
    return "TC3-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s" % (
        secret_id,
        scope,
        ";".join(signed_headers),
        signature,
    )


class TencentConfig:
    def __init__(self, secret_id: str, secret_key: str, region: str):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region


def tc_request(action: str, params: dict, cfg: TencentConfig, timeout: int = 60) -> dict:
    """调一次腾讯云 API，返回 Response 对象。出错时抛 RuntimeError。"""
    # 与官方 SDK 一致：默认 json.dumps（ensure_ascii=True），body 全 ASCII，
    # 这样不带 charset 的 content-type 不会有编码歧义。
    payload = json.dumps(params).encode("utf-8")
    timestamp = int(time.time())
    auth = tc3_authorization(
        cfg.secret_id, cfg.secret_key, TENCENT_SERVICE, TENCENT_HOST, action, payload, timestamp
    )
    req = urllib.request.Request("https://" + TENCENT_HOST + "/", data=payload, method="POST")
    req.add_header("Authorization", auth)
    req.add_header("Content-Type", TENCENT_CONTENT_TYPE)
    req.add_header("X-TC-Action", action)
    req.add_header("X-TC-Version", TENCENT_VERSION)
    req.add_header("X-TC-Timestamp", str(timestamp))
    if cfg.region:
        req.add_header("X-TC-Region", cfg.region)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTP {e.code}：{detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络不可达：{e.reason}")
    except Exception as e:
        raise RuntimeError(f"请求失败：{e}")

    resp = body.get("Response", {})
    err = resp.get("Error")
    if err:
        raise RuntimeError(f"{err.get('Code')}：{err.get('Message')}")
    return resp


def _pick(d: dict, *names, default=None):
    """大小写不敏感地取字段。

    腾讯云不同接口/版本对 ResultDetail 里的键名大小写不完全一致，
    硬写一种拼法很容易在读响应时炸掉。这里两种都认。
    """
    low = {str(k).lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return default


def tencent_create_task(b64_data: str, raw_len: int, engine: str, cfg: TencentConfig) -> str:
    resp = tc_request(
        "CreateRecTask",
        {
            "EngineModelType": engine,
            "ChannelNum": 1,
            # 开启说话人分离。编号是声学聚类，不是角色。
            "SpeakerDiarization": 1,
            # 16k 引擎不支持指定人数，只能用 0（自动）
            "SpeakerNumber": 0,
            # 2 = 句子 + 词级时间戳 + 语速 + 标点
            "ResTextFormat": 2,
            "SourceType": 1,
            "Data": b64_data,
            "DataLen": raw_len,
        },
        cfg,
    )
    task_id = _pick(resp.get("Data", {}) or {}, "TaskId", "taskId")
    if not task_id:
        raise RuntimeError("提交任务后没拿到 TaskId")
    return str(task_id)


def tencent_poll(task_id: str, cfg: TencentConfig, wait_max: float = 900.0):
    """轮询识别结果。返回原始 Data 对象。"""
    interval = 3.0
    deadline = time.time() + wait_max
    while time.time() < deadline:
        resp = tc_request("DescribeTaskStatus", {"TaskId": task_id}, cfg)
        data = resp.get("Data", {}) or {}
        status = _pick(data, "Status", "TaskStatus")
        if status == 2:
            return data, resp
        if status == 3:
            raise RuntimeError(f"识别失败：{_pick(resp, 'ErrorMsg') or _pick(data, 'ErrorMsg') or '未知原因'}")
        time.sleep(interval)
        interval = min(interval * 1.4, 8.0)
    raise RuntimeError("等待识别结果超时（已等 15 分钟）")


def rows_from_tencent(data: dict, offset_sec: float, chunk_idx: int, audio, sr, get_slice):
    """把腾讯云的 ResultDetail 转成统一的 jsonl 行。

    先认结构化结果（ResultDetail），认不到再退回整段纯文本（Result），
    这样至少不会一个字段对不上就整段丢失。
    """
    detail = _pick(data, "ResultDetail") or []
    rows = []
    if isinstance(detail, list) and detail:
        for item in detail:
            text = (_pick(item, "FinalSentence", "finalSentence")
                    or _pick(item, "SliceSentence", "sliceSentence") or "").strip()
            if not text:
                continue
            st = _pick(item, "StartTime", "startTime")
            et = _pick(item, "EndTime", "endTime")
            # 腾讯云给的是毫秒
            start = offset_sec + (float(st) / 1000.0 if st is not None else 0.0)
            end = offset_sec + (float(et) / 1000.0 if et is not None else start - offset_sec)
            spk = _pick(item, "SpeakerId", "speakerId")
            dur = max(end - start, 1e-6)
            seg = get_slice(audio, sr, start, end)
            row = {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "speaker": int(spk) if spk is not None else None,
                "chunk": chunk_idx,
                "avg_logprob": None,
                "no_speech_prob": None,
                "compression_ratio": None,
                "temperature": None,
                "rms_dbfs": dbfs(seg, "rms"),
                "peak_dbfs": dbfs(seg, "peak"),
                "chars_per_sec": round(len(text) / dur, 1),
            }
            speed = _pick(item, "SpeechSpeed", "speechSpeed")
            if speed is not None:
                row["speech_speed"] = speed
            words = _pick(item, "Words", "words")
            if isinstance(words, list) and words:
                row["words"] = [
                    {
                        "start": round(offset_sec + float(_pick(w, "StartTime", default=0)) / 1000.0, 2),
                        "end": round(offset_sec + float(_pick(w, "EndTime", default=0)) / 1000.0, 2),
                        "word": _pick(w, "Word", default=""),
                    }
                    for w in words
                ]
            rows.append(row)
        return rows

    plain = _pick(data, "Result") or ""
    if isinstance(plain, str) and plain.strip():
        rows.append({
            "start": round(offset_sec, 2),
            "end": round(offset_sec, 2),
            "text": plain.strip(),
            "speaker": None,
            "chunk": chunk_idx,
            "avg_logprob": None, "no_speech_prob": None,
            "compression_ratio": None, "temperature": None,
            "rms_dbfs": None, "peak_dbfs": None, "chars_per_sec": None,
        })
    return rows


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="语音转写：本地离线（faster-whisper）或腾讯云（含说话人分离）"
    )
    ap.add_argument("audio", help="音频文件路径")
    ap.add_argument("--out", default="./output", help="输出目录（默认 ./output）")
    ap.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "local", "tencent"],
        help="auto=有腾讯云密钥就用云端，否则本地（默认）；local=强制本地，绝不上传；"
        "tencent=强制云端，没密钥就报错",
    )
    ap.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="本地模型大小。默认 small（约 500MB）。"
        "若逐字稿里出现大量 [听不清]，换 medium 重跑（约 1.5GB）",
    )
    ap.add_argument(
        "--model-dir",
        default=None,
        help="本地模型目录（含 model.bin / config.json 等）。"
        "指定后忽略 --model。用于镜像自动下载失败时的手动兜底。",
    )
    ap.add_argument("--lang", default="zh", help="语言，默认 zh")
    ap.add_argument("--device", default="cpu", help="cpu 或 cuda，默认 cpu")
    ap.add_argument("--compute", default="int8", help="int8 / float16 / float32")
    ap.add_argument(
        "--words", action="store_true",
        help="本地路径下额外输出词级时间戳。体积会明显变大，默认关闭。",
    )
    ap.add_argument(
        "--codec", default="opus", choices=["opus", "mp3"],
        help="云端提交时的编码格式，默认 opus（同样体积下音质更好，更容易塞进 5MB 单次限额）",
    )
    ap.add_argument(
        "--tencent-region", default=None,
        help="腾讯云地域，默认取 TENCENTCLOUD_REGION，再默认 ap-guangzhou",
    )
    ap.add_argument(
        "--dump-raw", action="store_true",
        help="把云端每次调用的原始响应写到 <out>/<文件名>.raw.json，便于核对字段",
    )
    args = ap.parse_args()

    # ---- 选路 ----
    secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID", "").strip()
    secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY", "").strip()
    region = (args.tencent_region
              or os.environ.get("TENCENTCLOUD_REGION", "").strip()
              or "ap-guangzhou")

    if args.provider == "tencent":
        if not (secret_id and secret_key):
            die(
                "指定了 --provider tencent，但没找到密钥。请先设置环境变量：\n"
                "    TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY"
            )
        provider = "tencent"
    elif args.provider == "local":
        provider = "local"
    else:
        provider = "tencent" if (secret_id and secret_key) else "local"

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        die(f"文件不存在：{audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED:
        die(
            f"不支持的格式：{audio_path.suffix}\n"
            f"支持的格式：{', '.join(sorted(SUPPORTED))}\n"
            "提示：微信导出的 .silk 语音需要先转成 wav/mp3 才能处理。"
        )

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    if provider == "tencent":
        rows, meta = run_tencent(args, audio_path, out_dir, stem, secret_id, secret_key, region)
    else:
        rows, meta = run_local(args, audio_path, out_dir, stem)

    if not rows:
        die(
            "没有识别出任何文字。\n"
            "请检查：音频是否真的有人声、音量是否过小、格式是否受支持。"
        )

    write_outputs(out_dir, stem, rows, meta)


def run_local(args, audio_path: Path, out_dir: Path, stem: str):
    check_deps_local()
    from faster_whisper import WhisperModel

    if args.model_dir:
        src = str(Path(args.model_dir).expanduser().resolve())
        if not Path(src).is_dir():
            die(f"模型目录不存在：{src}")
        print(f"[1/4] 从本地目录加载模型：{src}")
    else:
        src = args.model
        print(f"[1/4] 加载模型 {args.model}（首次运行需下载，请耐心等待）...")
    t0 = time.time()
    try:
        model = WhisperModel(src, device=args.device, compute_type=args.compute)
    except Exception as e:
        die(
            f"模型加载失败：{e}\n\n"
            "已知原因：hf-mirror 会把 model.bin 重定向到 xet 存储，"
            "huggingface_hub 走这条路会下成 0 字节（报 'incomplete'）。\n"
            "兜底办法：用 curl 手动下载模型文件，再用 --model-dir 指定本地目录。\n"
            "  curl -sL -o <dir>/model.bin "
            "https://hf-mirror.com/Systran/faster-whisper-base/resolve/main/model.bin"
        )
    print(f"      模型就绪（{time.time() - t0:.1f}s）")

    print("[2/4] 读取音频（用于音量）...")
    audio, sr = decode_mono(audio_path)
    if audio is None:
        print("      未能读取音频，音量字段将留空（不影响转写）")
    else:
        print(f"      已读取 {audio.size / sr:.1f}s 音频")

    print(f"[3/4] 转写中：{audio_path.name}")
    t1 = time.time()
    try:
        segments, info = model.transcribe(
            str(audio_path),
            language=args.lang,
            vad_filter=True,
            beam_size=5,
            word_timestamps=args.words,
        )
    except Exception as e:
        die(f"转写失败：{e}\n可能原因：音频损坏、编码不支持，或文件不是有效音频。")

    print(f"      检测语言：{info.language}（置信度 {info.language_probability:.2f}）")
    print(f"      音频时长：{fmt_ts(info.duration)}")

    rows = []
    low = 0
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        if not text:
            continue
        dur = max(seg.end - seg.start, 1e-6)
        chunk = slice_at(audio, sr, seg.start, seg.end)
        row = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
            "speaker": None,
            "chunk": 0,
            # —— 以下为模型原始输出，不做换算 ——
            "avg_logprob": round(seg.avg_logprob, 4) if seg.avg_logprob is not None else None,
            "no_speech_prob": round(seg.no_speech_prob, 4) if seg.no_speech_prob is not None else None,
            "compression_ratio": round(seg.compression_ratio, 3) if seg.compression_ratio is not None else None,
            "temperature": seg.temperature,
            # —— 以下为客观派生量 ——
            "rms_dbfs": dbfs(chunk, "rms"),
            "peak_dbfs": dbfs(chunk, "peak"),
            "chars_per_sec": round(len(text) / dur, 1),
        }
        if args.words and seg.words:
            row["words"] = [
                {
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "word": w.word,
                    "prob": round(w.probability, 3) if w.probability is not None else None,
                }
                for w in seg.words
            ]
        if seg.avg_logprob is not None and seg.avg_logprob < AVG_LOGPROB_FLOOR:
            low += 1
        rows.append(row)
        if i % 20 == 0:
            print(f"      ... 已处理 {i} 段（当前 {fmt_ts(seg.end)}）")

    meta = {
        "provider": "local",
        "duration": info.duration,
        "language": info.language,
        "low_conf": low,
        "chunked": False,
        "audio_read": audio is not None,
        "cloud_note": None,
    }
    return rows, meta


def run_tencent(args, audio_path: Path, out_dir: Path, stem: str,
                secret_id: str, secret_key: str, region: str):
    cfg = TencentConfig(secret_id, secret_key, region)
    engine = LANG_TO_ENGINE.get(args.lang, "16k_zh")

    print(f"[1/4] 云端转写：腾讯云 {TENCENT_HOST}（地域 {region}，引擎 {engine}）")
    print("      注意：音频会上传到腾讯云处理。不想上传请用 --provider local。")

    print("[2/4] 读数并编码...")
    audio, sr = decode_mono(audio_path)
    if audio is None:
        size = audio_path.stat().st_size
        if b64_len(size) > TENCENT_HARD_LIMIT:
            die(
                "读不出音频（缺 PyAV），原始文件又超过腾讯云直传的 5MB 限额，无法提交。\n"
                "请执行：pip install av，然后重跑。"
            )
        print("      ⚠ 读不出音频，无法压缩也无法算音量。将按原文件提交。")
        print("        如果原文件采样率不是 16k，识别效果可能变差。")
        chunks = [(0, 0.0, None, None, None)]
    else:
        duration = audio.size / sr
        probe = out_dir / f".{stem}.probe.{args.codec}"
        plan = plan_cloud(audio, sr, args.codec, probe, TENCENT_SAFE_LIMIT)
        if plan is None:
            die(f"音频编码失败（{args.codec}）。请改用 --codec mp3 重试。")
        mode, bitrate, chunk_sec, probe_bytes = plan
        try:
            probe.unlink()  # 试编码用的临时文件，用完即删
        except Exception:
            pass
        print(f"      时长 {fmt_ts(duration)}；{args.codec} {bitrate}kbps 实测整段 "
              f"{probe_bytes / 1048576:.2f}MB")
        if mode == "single":
            chunks = [(0, 0.0, None, args.codec, bitrate)]
            print("      单段提交 —— 说话人编号全局一致")
        else:
            chunks = []
            t, idx = 0.0, 0
            while t < duration - 0.5:
                end = min(duration, t + chunk_sec)
                chunks.append((idx, t, end - t, args.codec, bitrate))
                t = end
                idx += 1
            print(f"      整段塞不进 5MB 单次限额，切成 {len(chunks)} 段"
                  f"（每段约 {chunk_sec / 60:.1f} 分钟）")
            print("      每段独立做说话人分离 —— **跨段编号不可比**")

    print(f"[3/4] 提交识别（共 {len(chunks)} 次请求）...")
    t1 = time.time()
    rows = []
    raw_log = []
    tmp_files = []

    def cleanup():
        """任何退出路径都要清掉中间编码文件——里面是原始录音，不能留在盘上。"""
        for f in tmp_files:
            try:
                f.unlink()
            except Exception:
                pass

    for idx, offset, dur, codec, br in chunks:
        if codec is None:
            send_path = audio_path
            raw_len = send_path.stat().st_size
        else:
            send_path = out_dir / f".{stem}.chunk{idx}.{codec}"
            seg_audio = slice_at(audio, sr, offset, offset + dur)
            if seg_audio is None or seg_audio.size == 0:
                continue
            ok = encode_audio(seg_audio, sr, codec, br, send_path)
            if not ok:
                cleanup()
                die(
                    f"第 {idx + 1} 段编码失败（{codec}）。\n"
                    "可改用 --codec mp3 重试；若仍失败，说明本机 PyAV 缺编码器。"
                )
            tmp_files.append(send_path)
            raw_len = send_path.stat().st_size

        b64_data = base64.b64encode(send_path.read_bytes()).decode("ascii")
        if len(b64_data) > TENCENT_HARD_LIMIT:
            cleanup()
            die(
                f"第 {idx + 1} 段 base64 后 {len(b64_data) / 1048576:.2f}MB，"
                f"超过腾讯云 5MB 限额。请用 --codec mp3 或缩短音频。"
            )
        print(f"      [{idx + 1}/{len(chunks)}] 提交 {fmt_ts(offset)} 起、"
              f"{len(b64_data) / 1048576:.2f}MB ...")
        try:
            task_id = tencent_create_task(b64_data, raw_len, engine, cfg)
            data, resp = tencent_poll(task_id, cfg)
        except RuntimeError as e:
            cleanup()
            die(f"云端转写失败：{e}")
        if args.dump_raw:
            raw_log.append({"chunk": idx, "task_id": task_id, "response": resp})
        got = rows_from_tencent(data, offset, idx, audio, sr, slice_at)
        print(f"            ✓ 得 {len(got)} 段")
        rows.extend(got)

    cleanup()

    if args.dump_raw and raw_log:
        raw_path = out_dir / f"{stem}.raw.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(raw_log, f, ensure_ascii=False, indent=2)
        print(f"      原始响应：{raw_path}")

    meta = {
        "provider": "tencent",
        "duration": audio.size / sr if audio is not None else None,
        "language": engine,
        "low_conf": 0,
        "chunked": len(chunks) > 1,
        "n_chunks": len(chunks),
        "audio_read": audio is not None,
        "cloud_note": f"{TENCENT_HOST}（{region}）",
        "elapsed": time.time() - t1,
    }
    return rows, meta


def write_outputs(out_dir: Path, stem: str, rows: list, meta: dict) -> None:
    jsonl_path = out_dir / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    md_path = out_dir / f"{stem}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# 逐字稿：{stem}\n\n")
        if meta.get("duration"):
            f.write(f"- 音频时长：{fmt_ts(meta['duration'])}\n")
        f.write(f"- 片段数：{len(rows)}\n")
        f.write(f"- 转写路径：{'腾讯云（上传处理）' if meta['provider'] == 'tencent' else '本地离线'}\n")
        if meta["provider"] == "local":
            f.write(f"- 低置信片段：{meta['low_conf']} 段（标记为 [听不清]）\n")
            f.write(f"- 检测语言：{meta['language']}\n")
        if meta.get("chunked"):
            f.write(
                f"- ⚠ 已切成 {meta.get('n_chunks')} 段分别识别："
                "**说话人编号只在同一段内可比，跨段不可比**\n"
            )
        if not meta.get("audio_read", True):
            f.write("- 音量：未读取到音频，本份不含音量数据\n")
        if meta["provider"] == "tencent":
            f.write("- 说明：本份含说话人编号，但编号只是声学聚类，不是角色\n")
            f.write("- 云端不返回模型内部置信度，故不含 avg_logprob 等字段\n")
        else:
            f.write("- **说话人未区分**，由上层 AI 推断后需用户确认\n")
        f.write("\n---\n\n")
        last_chunk = 0
        for r in rows:
            if r.get("chunk", 0) != last_chunk:
                last_chunk = r.get("chunk", 0)
                f.write(f"\n<!-- 第 {last_chunk + 1} 段切片开始（此段内说话人编号重新计数） -->\n\n")
            ts = f"{fmt_ts(r['start'])}–{fmt_ts(r['end'])}"
            tag = ""
            if r.get("avg_logprob") is not None and r["avg_logprob"] < AVG_LOGPROB_FLOOR:
                tag = " `[听不清]`"
            spk = r.get("speaker")
            spk_s = f" **说话人{spk}**：" if spk is not None else ""
            f.write(f"**[{ts}]**{tag}{spk_s} {r['text']}\n\n")

    print(f"[4/4] 完成")
    print(f"      逐字稿：{md_path}")
    print(f"      结构化：{jsonl_path}")
    if meta["provider"] == "local" and meta.get("low_conf"):
        pct = meta["low_conf"] / len(rows) * 100
        print(f"\n[注意] {meta['low_conf']} 段置信度偏低（{pct:.1f}%）。")
        print("       这些片段在分析时会被标为 [听不清]，不能作为结论依据。")
        if pct > 20:
            print("       占比偏高，建议用 --model medium 重跑一次。")
    if meta.get("chunked"):
        print("\n[注意] 音频被切成多段识别，说话人编号跨段不可比。")
        print("       分析前必须按对话内容做一次全局角色确认。")


if __name__ == "__main__":
    main()
