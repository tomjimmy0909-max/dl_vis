"""「数据与预处理」侧栏：仅从本列表向画布拖拽预处理节点（与主算子调色板分离）。"""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from dl_vis.model.node_types import preproc_palette_types
from dl_vis.ui import locale_zh as ZH
from dl_vis.ui.palette import PaletteList


class DataProcPanel(QWidget):
    """拖列表面 + 说明；参数在右侧 Dock（选中节点后）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        hint = QLabel(ZH.TAB_DATA_PROC_HINT)
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self._list = PaletteList(type_keys=preproc_palette_types())
        lay.addWidget(self._list, 1)
