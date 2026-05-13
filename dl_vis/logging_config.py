"""日志与诊断：便于排查闪退（含 Qt 原生层）。"""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

_LOG_MARK = "_dl_vis_logging_configured"


def setup_logging(level: int | None = None) -> None:
    root = logging.getLogger()
    if getattr(root, _LOG_MARK, False):
        return

    if level is None:
        lv_name = os.environ.get("DL_VIS_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, lv_name, logging.INFO)

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    log_file = os.environ.get("DL_VIS_LOG_FILE")
    if not log_file and os.environ.get("DL_VIS_DEBUG", "").strip() in ("1", "true", "TRUE", "yes", "YES"):
        log_file = str(Path(tempfile.gettempdir()) / "dl_vis.log")

    file_err: str | None = None
    if log_file:
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(fmt))
            fh.setLevel(logging.DEBUG)
            handlers.append(fh)
        except OSError as e:
            file_err = str(e)

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    setattr(root, _LOG_MARK, True)

    lg = logging.getLogger("dl_vis")
    if file_err:
        lg.warning("无法创建日志文件 %s：%s", log_file, file_err)
    elif log_file:
        lg.info("文件日志：%s", log_file)


def enable_faulthandler() -> None:
    """子线程崩溃时尽量把 Python 栈打到 stderr（原生段错误仍需调试器）。"""
    try:
        faulthandler.enable(all_threads=True)
    except RuntimeError:
        pass


def install_excepthook(log: logging.Logger | None = None) -> Callable[..., None]:
    """未捕获异常写入日志后再交给默认处理（仍会终止进程）。"""
    log = log or logging.getLogger("dl_vis")
    prev = sys.excepthook

    def _hook(exc_type, exc, tb) -> None:
        log.critical("未捕获异常", exc_info=(exc_type, exc, tb))
        try:
            from dl_vis.error_report import log_report_location, write_error_report

            path = write_error_report(exc_type, exc, tb, source="sys.excepthook_main")
            log_report_location(log, path, source="sys.excepthook_main")
        except Exception:
            log.exception("DL_VIS_ERROR_REPORT 主线程写入失败")
        prev(exc_type, exc, tb)

    sys.excepthook = _hook
    return prev


def install_thread_excepthook(log: logging.Logger | None = None) -> None:
    """子线程未捕获异常同样落盘 JSON（Python 3.8+）。"""
    import threading

    log = log or logging.getLogger("dl_vis")
    prev = getattr(threading, "excepthook", None)
    if prev is None:
        return

    def _hook(args: threading.ExceptHookArgs) -> None:  # type: ignore[attr-defined]
        log.critical(
            "未捕获线程异常 thread=%s",
            getattr(args.thread, "name", str(args.thread)),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        try:
            from dl_vis.error_report import log_report_location, write_error_report

            path = write_error_report(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                source="threading.excepthook",
                context_extra={"thread_name": getattr(args.thread, "name", "")},
            )
            log_report_location(log, path, source="threading.excepthook")
        except Exception:
            log.exception("DL_VIS_ERROR_REPORT 子线程写入失败")
        prev(args)

    threading.excepthook = _hook  # type: ignore[assignment]


def install_qt_message_handler(log: logging.Logger | None = None) -> Callable[..., None] | None:
    """将 Qt 的 qWarning/qCritical 等写入 Python 日志。"""
    try:
        from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:
        return None

    log = log or logging.getLogger("qt")

    def _handler(mode: QtMsgType, _context: object, message: str) -> None:
        msg = message.strip()
        if mode == QtMsgType.QtDebugMsg:
            log.debug("%s", msg)
        elif mode == QtMsgType.QtInfoMsg:
            log.info("%s", msg)
        elif mode == QtMsgType.QtWarningMsg:
            log.warning("%s", msg)
        elif mode == QtMsgType.QtCriticalMsg:
            log.error("%s", msg)
        elif mode == QtMsgType.QtFatalMsg:
            log.critical("%s", msg)
        else:
            log.info("%s", msg)

    prev = qInstallMessageHandler(_handler)
    return prev
