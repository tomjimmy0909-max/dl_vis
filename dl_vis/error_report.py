"""结构化异常报告：固定 Schema、落盘路径，便于自动化采集与 AI 解析。

当程序发生未捕获异常时，将异常的详细信息（类型、消息、堆栈、运行环境）
以 JSON 格式写入临时目录中的错误报告文件，方便后续排查。
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ERROR_REPORT_FORMAT = "dl_vis.error_report/v1"  # 报告格式版本
ENV_REPORT_DIR = "DL_VIS_ERROR_REPORT_DIR"       # 可通过环境变量自定义报告目录


def report_dir() -> Path:
    """获取错误报告目录（可被环境变量 DL_VIS_ERROR_REPORT_DIR 覆盖）。"""
    raw = os.environ.get(ENV_REPORT_DIR, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(tempfile.gettempdir()) / "dl_vis_error_reports"


def last_error_path() -> Path:
    """最新一次错误报告的 JSON 文件路径（覆盖写入）。"""
    return report_dir() / "dl_vis_last_error.json"


def error_events_path() -> Path:
    """所有历史错误事件追加到 JSONL 文件。"""
    return report_dir() / "dl_vis_error_events.jsonl"


def _optional_versions() -> dict[str, str]:
    """获取可选运行时的版本信息（Qt、PyTorch 等）。"""
    out: dict[str, str] = {}
    try:
        from PyQt6.QtCore import qVersion

        out["qt_runtime"] = qVersion()
    except Exception:
        pass
    try:
        import torch

        out["torch"] = getattr(torch, "__version__", "?")
    except Exception:
        pass
    return out


def build_context(*, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """构建错误报告的上下文信息（版本、平台、当前工作目录等）。"""
    ctx: dict[str, Any] = {
        "dl_vis_version": _package_version(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    ctx.update(_optional_versions())
    if extra:
        ctx.update(dict(extra))
    return ctx


def _package_version() -> str:
    """读取 dl_vis 包的版本号。"""
    try:
        from dl_vis import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def build_report(
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    tb: Any,
    *,
    source: str,
    context_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造完整的错误报告字典（JSON 兼容）。"""
    et = exc_type.__name__ if exc_type is not None else "NoneType"
    msg = str(exc) if exc is not None else ""
    tb_text = ""
    if exc_type is not None and tb is not None:
        tb_text = "".join(traceback.format_exception(exc_type, exc, tb))
    return {
        "format": ERROR_REPORT_FORMAT,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exception_type": et,
        "exception_message": msg,
        "traceback": tb_text,
        "context": build_context(extra=context_extra),
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def write_error_report(
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    tb: Any,
    *,
    source: str,
    context_extra: Mapping[str, Any] | None = None,
) -> Path | None:
    """写入 ``dl_vis_last_error.json`` 并追加一行 ``dl_vis_error_events.jsonl``；失败则返回 None。"""
    try:
        rep = build_report(exc_type, exc, tb, source=source, context_extra=context_extra)
        last = last_error_path()
        events = error_events_path()
        _atomic_write_json(last, rep)
        append_jsonl(events, rep)
        return last
    except Exception:
        return None


def write_error_report_from_exc(
    exc_info: tuple[type[BaseException], BaseException, Any] | None,
    *,
    source: str,
    context_extra: Mapping[str, Any] | None = None,
) -> Path | None:
    if exc_info is None or exc_info[0] is None:
        return None
    return write_error_report(exc_info[0], exc_info[1], exc_info[2], source=source, context_extra=context_extra)


def log_report_location(log: Any, path: Path | None, *, source: str) -> None:
    """一行固定前缀，便于 grep / 工具链抓取。"""
    if path is None:
        log.error("DL_VIS_ERROR_REPORT write_failed source=%s", source)
        return
    log.error(
        "DL_VIS_ERROR_REPORT written source=%s path=%s format=%s",
        source,
        str(path.resolve()),
        ERROR_REPORT_FORMAT,
    )
