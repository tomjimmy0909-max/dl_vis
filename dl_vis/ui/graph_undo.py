"""基于 QUndoStack 的 GraphDocument 快照撤销/重做。"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from PyQt6.QtGui import QUndoCommand

if TYPE_CHECKING:
    from dl_vis.ui.canvas_widget import CanvasWidget


class DocStateCommand(QUndoCommand):
    """保存整图快照；undo/redo 时替换 Canvas 上的 GraphDocument 并重绘。"""

    def __init__(self, canvas: CanvasWidget, before: dict, after: dict, description: str) -> None:
        super().__init__(description)
        self._canvas = canvas
        self._before = copy.deepcopy(before)
        self._after = copy.deepcopy(after)

    def undo(self) -> None:
        self._canvas.apply_doc_snapshot(self._before)

    def redo(self) -> None:
        self._canvas.apply_doc_snapshot(self._after)
