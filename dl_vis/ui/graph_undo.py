"""基于 QUndoStack 的 GraphDocument 快照撤销/重做。"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from PyQt6.QtGui import QUndoCommand

if TYPE_CHECKING:
    from dl_vis.ui.canvas_widget import CanvasWidget


class DocStateCommand(QUndoCommand):
    """保存整图快照的撤销/重做命令。

    每次用户操作（添加/删除节点、连线、移动等）时，
    记录前后两份完整的图状态快照（dict），
    undo/redo 时通过 CanvasWidget.apply_doc_snapshot() 恢复到对应状态。
    """

    def __init__(self, canvas: CanvasWidget, before: dict, after: dict, description: str) -> None:
        super().__init__(description)
        self._canvas = canvas
        self._before = copy.deepcopy(before)  # 操作前的完整图快照
        self._after = copy.deepcopy(after)    # 操作后的完整图快照

    def undo(self) -> None:
        """撤销：恢复到操作前的状态。"""
        self._canvas.apply_doc_snapshot(self._before)

    def redo(self) -> None:
        """重做：恢复到操作后的状态。"""
        self._canvas.apply_doc_snapshot(self._after)
