"""后台训练线程：避免阻塞 UI。"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from PyQt6.QtCore import QThread, pyqtSignal

from dl_vis.logic.graph_executor import GraphExecutor

if TYPE_CHECKING:
    from dl_vis.model.graph_document import GraphDocument


DataMode = Literal["synthetic", "npy", "csv", "graph"]


@dataclass
class TrainingJobConfig:
    epochs: int
    lr: float = 1e-3
    mode: DataMode = "synthetic"
    x_path: str = ""
    y_path: str = ""
    csv_path: str = ""
    csv_skip_header: bool = False
    channels: int = 3
    height: int = 224
    width: int = 224


class TrainingWorker(QThread):
    """在子线程中运行 ``GraphExecutor`` 训练；通过信号回传进度与结果。"""

    epoch_loss = pyqtSignal(int, float)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, doc: GraphDocument, cfg: TrainingJobConfig, parent: QThread | None = None) -> None:
        super().__init__(parent)
        self._doc = doc
        self._cfg = cfg

    def run(self) -> None:  # type: ignore[override]
        try:
            ex = GraphExecutor(self._doc)

            def on_ep(ep: int, lv: float) -> None:
                self.epoch_loss.emit(ep, lv)

            if self._cfg.mode == "graph":
                losses = ex.train_from_graph_dataset(
                    epochs=self._cfg.epochs,
                    lr=self._cfg.lr,
                    skip_trailing_softmax=True,
                    on_epoch=on_ep,
                )
                self.finished_ok.emit(losses)
                return

            if self._cfg.mode == "synthetic":
                losses = ex.train_synthetic(
                    epochs=self._cfg.epochs,
                    lr=self._cfg.lr,
                    skip_trailing_softmax=True,
                    on_epoch=on_ep,
                )
                self.finished_ok.emit(losses)
                return

            if self._cfg.mode == "npy":
                x, y = ex.load_npy_pair(self._cfg.x_path, self._cfg.y_path)
                losses = ex.train_with_arrays(
                    x,
                    y,
                    epochs=self._cfg.epochs,
                    lr=self._cfg.lr,
                    skip_trailing_softmax=True,
                    on_epoch=on_ep,
                )
                self.finished_ok.emit(losses)
                return

            if self._cfg.mode == "csv":
                x, y = ex.load_csv_nchw_labels(
                    self._cfg.csv_path,
                    channels=self._cfg.channels,
                    height=self._cfg.height,
                    width=self._cfg.width,
                    skip_header_row=self._cfg.csv_skip_header,
                )
                losses = ex.train_with_arrays(
                    x,
                    y,
                    epochs=self._cfg.epochs,
                    lr=self._cfg.lr,
                    skip_trailing_softmax=True,
                    on_epoch=on_ep,
                )
                self.finished_ok.emit(losses)
                return

            self.failed.emit("未知的数据模式。")
        except Exception as e:  # noqa: BLE001 — 线程边界统一转字符串
            log = logging.getLogger("dl_vis.training")
            log.exception("训练线程失败 mode=%s", self._cfg.mode)
            try:
                from dl_vis.error_report import log_report_location, write_error_report_from_exc

                path = write_error_report_from_exc(
                    sys.exc_info(),
                    source="training_worker",
                    context_extra={
                        "training_mode": self._cfg.mode,
                        "epochs": self._cfg.epochs,
                        "lr": self._cfg.lr,
                    },
                )
                log_report_location(log, path, source="training_worker")
            except Exception:
                log.exception("DL_VIS_ERROR_REPORT 训练线程写入失败")
            self.failed.emit(str(e))
