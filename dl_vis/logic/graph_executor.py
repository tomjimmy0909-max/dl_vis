"""GraphExecutor：只读 GraphDocument，封装构建模型、前向与训练（对齐 DESIGN 9.1 雏形）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from dl_vis.logic import runtime_torch as rt

if TYPE_CHECKING:
    import torch

    from dl_vis.model.graph_document import GraphDocument


class GraphExecutor:
    """
    执行器持有文档引用，负责构建 ``nn.Sequential``、前向与 CE 训练。
    拓扑仍以 ``GraphDocument`` 为权威；本类不修改文档。
    """

    def __init__(self, doc: GraphDocument) -> None:
        self._doc = doc

    @property
    def document(self) -> GraphDocument:
        return self._doc

    def build_model(self, *, skip_trailing_softmax: bool = False) -> Any:
        return rt.build_sequential(self._doc, skip_trailing_softmax=skip_trailing_softmax)

    def dummy_forward(self) -> dict[str, Any]:
        return rt.dummy_forward(self._doc)

    def train_synthetic(
        self,
        *,
        epochs: int,
        lr: float = 1e-3,
        skip_trailing_softmax: bool = True,
        on_epoch: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        return rt.train_synthetic_ce(
            self._doc,
            epochs=epochs,
            lr=lr,
            skip_trailing_softmax=skip_trailing_softmax,
            on_epoch=on_epoch,
        )

    def load_npy_pair(self, x_path: str, y_path: str) -> tuple["torch.Tensor", "torch.Tensor"]:
        return rt.load_npy_training_pair(x_path, y_path)

    def load_csv_nchw_labels(
        self,
        path: str,
        *,
        channels: int,
        height: int,
        width: int,
        skip_header_row: bool = False,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        return rt.load_csv_nchw_labels(
            path, channels=channels, height=height, width=width, skip_header_row=skip_header_row
        )

    def train_with_arrays(
        self,
        x: "torch.Tensor",
        y: "torch.Tensor",
        *,
        epochs: int,
        lr: float = 1e-3,
        skip_trailing_softmax: bool = True,
        on_epoch: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        return rt.train_with_tensors(
            self._doc,
            x,
            y,
            epochs=epochs,
            lr=lr,
            skip_trailing_softmax=skip_trailing_softmax,
            on_epoch=on_epoch,
        )

    def train_from_graph_dataset(
        self,
        *,
        epochs: int,
        lr: float = 1e-3,
        skip_trailing_softmax: bool = True,
        on_epoch: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        """使用 Dataset→Input 绑定的路径加载数据并训练（与菜单「图上数据集」一致）。"""
        return rt.train_from_graph_dataset(
            self._doc,
            epochs=epochs,
            lr=lr,
            skip_trailing_softmax=skip_trailing_softmax,
            on_epoch=on_epoch,
        )
