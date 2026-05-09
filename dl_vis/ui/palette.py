"""可拖拽节点类型列表（拖到画布创建节点）；延迟显示中文结构说明。"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QEvent, QPoint, QRectF, Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem, QToolTip

from dl_vis.model.node_types import NODE_PALETTE_ZH, palette_types
from dl_vis.ui.canvas_widget import MIME_NODE_TYPE

# QListWidgetItem：存储英文类型键（序列化用）与说明全文
ROLE_TYPE_KEY = Qt.ItemDataRole.UserRole
ROLE_DOC_ZH = Qt.ItemDataRole.UserRole + 1

# 悬停多久后弹出说明（毫秒）；文档 tooltip 显示时长
TOOLTIP_DELAY_MS = 1200
TOOLTIP_SHOW_DURATION_MS = 45000


class PaletteList(QListWidget):
    """Drag-only 列表；MIME 携带英文节点类型；中文标签 + 延迟结构说明。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self._tip_timer = QTimer(self)
        self._tip_timer.setSingleShot(True)
        self._tip_timer.timeout.connect(self._show_pending_doc_tooltip)

        self._hover_item: QListWidgetItem | None = None

        for type_key in palette_types():
            pair = NODE_PALETTE_ZH.get(type_key)
            label = pair[0] if pair else type_key
            doc = pair[1] if pair else ""
            item = QListWidgetItem(label)
            item.setData(ROLE_TYPE_KEY, type_key)
            item.setData(ROLE_DOC_ZH, doc)
            item.setToolTip("")  # 禁用即时原生 tooltip，改由定时器展示长说明
            self.addItem(item)

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        vp_pos = self.viewport().mapFrom(self, event.pos())
        it = self.itemAt(vp_pos)
        if it is not self._hover_item:
            self._hover_item = it
            self._tip_timer.stop()
            QToolTip.hideText()
            if it is not None:
                self._tip_timer.start(TOOLTIP_DELAY_MS)

    def _show_pending_doc_tooltip(self) -> None:
        vp = self.viewport()
        pos_vp = vp.mapFromGlobal(QCursor.pos())
        item = self.itemAt(pos_vp)
        if item is None:
            return
        doc = item.data(ROLE_DOC_ZH)
        if not doc:
            return
        rect = self.visualItemRect(item)
        hot_rect = rect.toRect() if isinstance(rect, QRectF) else rect
        global_anchor = vp.mapToGlobal(hot_rect.topLeft() + QPoint(8, hot_rect.height() // 2))
        QToolTip.showText(global_anchor, doc, vp, hot_rect, TOOLTIP_SHOW_DURATION_MS)

    def leaveEvent(self, event: QEvent) -> None:
        self._tip_timer.stop()
        QToolTip.hideText()
        self._hover_item = None
        super().leaveEvent(event)

    def startDrag(self, supported_actions: Qt.DropAction) -> None:
        self._tip_timer.stop()
        QToolTip.hideText()
        super().startDrag(supported_actions)

    def mimeData(self, items: list[QListWidgetItem]):
        md = super().mimeData(items)
        if items:
            type_key = items[0].data(ROLE_TYPE_KEY)
            if type_key is None:
                type_key = items[0].text()
            else:
                type_key = str(type_key)
            md.setData(MIME_NODE_TYPE, QByteArray(type_key.encode("utf-8")))
        return md

    def supportedDragActions(self) -> Qt.DropAction:
        return Qt.DropAction.CopyAction
