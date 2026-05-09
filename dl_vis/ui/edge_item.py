"""节点之间的连线图形项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsPathItem

if TYPE_CHECKING:
    from dl_vis.ui.node_item import NodeItem


class EdgeItem(QGraphicsPathItem):
    def __init__(self, edge_id: str, src_id: str, dst_id: str) -> None:
        super().__init__()
        self.edge_id = edge_id
        self.src_id = src_id
        self.dst_id = dst_id
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(-1)
        self._pen_normal = QPen(QColor(90, 90, 90), 2, Qt.PenStyle.SolidLine)
        self._pen_sel = QPen(QColor(0, 120, 215), 3, Qt.PenStyle.SolidLine)
        self.setPen(self._pen_normal)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsPathItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setPen(self._pen_sel if self.isSelected() else self._pen_normal)
        return result

    def refresh_geometry(self, src_item: NodeItem | None, dst_item: NodeItem | None) -> None:
        self.prepareGeometryChange()
        if src_item is None or dst_item is None:
            self.setPath(QPainterPath())
            return
        p0 = src_item.output_port_scene_center()
        p3 = dst_item.input_port_scene_center()
        path = QPainterPath(p0)
        dx = max(40.0, abs(p3.x() - p0.x()) * 0.5)
        c1 = QPointF(p0.x() + dx, p0.y())
        c2 = QPointF(p3.x() - dx, p3.y())
        path.cubicTo(c1, c2, p3)
        self.setPath(path)
