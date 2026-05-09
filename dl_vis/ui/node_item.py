"""画布上的节点图形项：矩形、标题、输入/输出端口。"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from PyQt6.QtWidgets import (
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

NODE_WIDTH = 140
NODE_HEIGHT = 70
PORT_RADIUS = 8


class NodeItem(QGraphicsObject):
    """可拖动节点；端口用于连线。"""

    moved = pyqtSignal(str, float, float)  # node_id, x, y scene pos of origin
    output_drag_started = pyqtSignal(str, QPointF)  # node_id, scene pos of line start
    output_drag_moved = pyqtSignal(QPointF)
    output_drag_finished = pyqtSignal(QPointF)

    def __init__(self, node_id: str, title: str, parent: QGraphicsObject | None = None) -> None:
        super().__init__(parent)
        self.node_id = node_id
        self.title = title
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self._dragging_wire = False

    def boundingRect(self) -> QRectF:
        # 端口圆绘制在 x<0 / x>NODE_WIDTH，必须在包围盒内，否则拖动时旧区域不会被正确刷新而产生残影
        pen_pad = 4.0
        return QRectF(
            -PORT_RADIUS - pen_pad,
            -pen_pad,
            NODE_WIDTH + 2 * PORT_RADIUS + 2 * pen_pad,
            NODE_HEIGHT + 2 * pen_pad,
        )

    def input_port_scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, NODE_HEIGHT / 2))

    def output_port_scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(NODE_WIDTH, NODE_HEIGHT / 2))

    def port_hit_scene(self, scene_pos: QPointF) -> str | None:
        """返回 'in' / 'out' / None。"""
        lp = self.mapFromScene(scene_pos)
        ir = QRectF(-PORT_RADIUS, NODE_HEIGHT / 2 - PORT_RADIUS, 2 * PORT_RADIUS, 2 * PORT_RADIUS)
        or_ = QRectF(NODE_WIDTH - PORT_RADIUS, NODE_HEIGHT / 2 - PORT_RADIUS, 2 * PORT_RADIUS, 2 * PORT_RADIUS)
        if ir.contains(lp):
            return "in"
        if or_.contains(lp):
            return "out"
        return None

    def paint(self, painter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        body = QRectF(0, 0, NODE_WIDTH, NODE_HEIGHT)
        pen = QPen(QColor(60, 60, 60), 2)
        if self.isSelected():
            pen.setColor(QColor(0, 120, 215))
            pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(245, 248, 252)))
        painter.drawRoundedRect(body, 6, 6)

        # ports
        painter.setBrush(QBrush(QColor(80, 160, 240)))
        painter.setPen(QPen(QColor(40, 100, 180)))
        cy = NODE_HEIGHT / 2
        painter.drawEllipse(QRectF(-PORT_RADIUS, cy - PORT_RADIUS, 2 * PORT_RADIUS, 2 * PORT_RADIUS))
        painter.drawEllipse(QRectF(NODE_WIDTH - PORT_RADIUS, cy - PORT_RADIUS, 2 * PORT_RADIUS, 2 * PORT_RADIUS))

        painter.setPen(QPen(QColor(30, 30, 30)))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(body.adjusted(10, 0, -10, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter), self.title)

    def itemChange(self, change, value):
        if change == QGraphicsObject.GraphicsItemChange.ItemPositionHasChanged:
            self.moved.emit(self.node_id, self.pos().x(), self.pos().y())
        return super().itemChange(change, value)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        port = self.port_hit_scene(event.scenePos())
        if port == "out" and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_wire = True
            self.output_drag_started.emit(self.node_id, self.output_port_scene_center())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_wire:
            self.output_drag_moved.emit(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_wire:
            self._dragging_wire = False
            self.output_drag_finished.emit(event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)
