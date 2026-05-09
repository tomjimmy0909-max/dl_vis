"""PyTorch 导出占位（第二阶段实现）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dl_vis.model.graph_document import GraphDocument


def export_to_torch_module(_document: GraphDocument) -> Any:
    """
    将 GraphDocument 转为可执行的 nn.Module 或生成源码。
    第一阶段：未实现，调用方应捕获或检查占位。
    """
    raise NotImplementedError("PyTorch 导出计划在第二阶段实现")


def export_stub_message() -> str:
    return "PyTorch 导出功能尚未实现（参见 docs/DESIGN.md 路线图）。"
