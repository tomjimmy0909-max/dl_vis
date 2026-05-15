"""实时摄像头：有界队列丢旧帧以降低端到端延迟。"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise ImportError("摄像头预览需要 OpenCV：pip install opencv-python") from e
    return cv2


class WebcamFrameQueue:
    """
    后台线程抓取 BGR 帧，``queue`` 最大 ``maxsize``：满时丢弃最旧帧再放入新帧。
    ``get_latest_timeout`` 便于 UI 线程非阻塞轮询。
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        queue_max: int = 2,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._cv2 = _import_cv2()
        self._device_index = int(device_index)
        self._queue_max = max(1, int(queue_max))
        self._width = int(width)
        self._height = int(height)
        self._q: queue.Queue[tuple[Any, float]] = queue.Queue(maxsize=self._queue_max)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: Any = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dlvis-webcam", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cv2 = self._cv2
        cap = cv2.VideoCapture(self._device_index)
        self._cap = cap
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            ts = time.time()
            try:
                self._q.put_nowait((frame, ts))
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait((frame, ts))
                except queue.Full:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def get_latest(self, *, timeout: float = 0.0) -> tuple[Any, float] | None:
        """返回 ``(bgr_frame, t_capture)`` 或超时 ``None``。"""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_to_latest(self) -> tuple[Any, float] | None:
        """清空队列只保留最后一帧（最低显示延迟）。"""
        last: tuple[Any, float] | None = None
        while True:
            try:
                last = self._q.get_nowait()
            except queue.Empty:
                break
        return last
