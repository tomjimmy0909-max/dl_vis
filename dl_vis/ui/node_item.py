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
    """可拖动节点；端口用于连线。

    每个 NodeItem 对应 GraphDocument 中的一个 GraphNode。
    提供三个交互区域：
    - 主体矩形：拖动移动、选中
    - 左侧蓝色圆点：输入端口（接受连线）
    - 右侧蓝色圆点：输出端口（拖出连线）
    """

    # 信号：通知画布节点位置变化、拖动结束、输出端口拖拽
    position_changed = pyqtSignal(str)  # node_id；画布据此写回 GraphDocument 并刷新连线
    live_drag_finished = pyqtSignal(str)  # 左键在节点主体释放（含拖动结束）；用于延迟提交撤销点
    output_drag_started = pyqtSignal(str, QPointF)  # node_id, scene pos of line start
    output_drag_moved = pyqtSignal(QPointF)
    output_drag_finished = pyqtSignal(QPointF)

    def __init__(self, node_id: str, title: str, *, accent: str | None = None, parent: QGraphicsObject | None = None) -> None:
        super().__init__(parent)
        self.node_id = node_id
        self.title = title
        self._accent = accent  # "dataset" 绿色；"dataproc" 浅琥珀（数据处理节点）
        # 启用可移动、可选中、几何变化追踪
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self._dragging_wire = False  # 是否正在从输出端口拖出连线

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
        """检测场景坐标是否落在某个端口上，返回 'in' / 'out' / None。"""
        lp = self.mapFromScene(scene_pos)
        # 左侧输入端口和右侧输出端口的点击区域
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
        if self._accent == "dataset":
            fill = QColor(222, 245, 222)
        elif self._accent == "dataproc":
            fill = QColor(255, 248, 220)
        else:
            fill = QColor(245, 248, 252)
        painter.setBrush(QBrush(fill))
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
            self.position_changed.emit(self.node_id)
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
        if event.button() == Qt.MouseButton.LeftButton:
            self.live_drag_finished.emit(self.node_id)
