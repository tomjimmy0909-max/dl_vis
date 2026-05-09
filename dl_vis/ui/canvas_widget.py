"""单页画布：场景、网格、节点/边同步、连线拖拽、删除。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDragEnterEvent, QDropEvent, QKeyEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
)

from dl_vis.model.graph_document import GraphDocument
from dl_vis.model.node_types import palette_label_zh
from dl_vis.ui.edge_item import EdgeItem
from dl_vis.ui import locale_zh as ZH
from dl_vis.ui.node_item import NODE_HEIGHT, NODE_WIDTH, NodeItem

MIME_NODE_TYPE = "application/x-dlvis-node-type"

if TYPE_CHECKING:
    pass


class CanvasWidget(QGraphicsView):
    """一个 Tab 对应一个 CanvasWidget + GraphDocument。"""

    selection_node_changed = pyqtSignal(object)  # GraphNode | None
    document_modified = pyqtSignal()
    status_message = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc = GraphDocument()
        self._node_items: dict[str, NodeItem] = {}
        self._edge_items: dict[str, EdgeItem] = {}
        self._temp_wire: QGraphicsLineItem | None = None
        self._wire_src_id: str | None = None

        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(QRectF(-2000, -2000, 4000, 4000))
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # MinimalViewportUpdate + 抗锯齿在 Windows 上易导致拖动残影，改用智能刷新
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setAcceptDrops(True)
        self.scale(1.0, 1.0)

        self._scene.selectionChanged.connect(self._on_scene_selection_changed)

    def graph_document(self) -> GraphDocument:
        return self._doc

    def set_graph_document(self, doc: GraphDocument) -> None:
        self._doc = doc
        self._rebuild_from_document()

    def _rebuild_from_document(self) -> None:
        self._scene.blockSignals(True)
        self._scene.clear()
        self._node_items.clear()
        self._edge_items.clear()
        self._clear_temp_wire()

        for gn in self._doc.iter_nodes():
            item = NodeItem(gn.id, palette_label_zh(gn.type))
            item.setPos(QPointF(gn.x, gn.y))
            item.moved.connect(self._on_node_moved)
            item.output_drag_started.connect(self._on_output_drag_started)
            item.output_drag_moved.connect(self._on_output_drag_moved)
            item.output_drag_finished.connect(self._on_output_drag_finished)
            self._scene.addItem(item)
            self._node_items[gn.id] = item

        for ge in self._doc.iter_edges():
            ei = EdgeItem(ge.id, ge.src_id, ge.dst_id)
            ei.refresh_geometry(self._node_items.get(ge.src_id), self._node_items.get(ge.dst_id))
            self._scene.addItem(ei)
            self._edge_items[ge.id] = ei

        self._scene.blockSignals(False)
        self._on_scene_selection_changed()

    def _clear_temp_wire(self) -> None:
        if self._temp_wire is not None:
            self._scene.removeItem(self._temp_wire)
            self._temp_wire = None
        self._wire_src_id = None

    def _on_node_moved(self, node_id: str, x: float, y: float) -> None:
        self._doc.update_node_position(node_id, x, y)
        self._refresh_edges_for(node_id)
        self.document_modified.emit()

    def _refresh_edges_for(self, node_id: str) -> None:
        for e in self._doc.edges.values():
            if e.src_id == node_id or e.dst_id == node_id:
                ei = self._edge_items.get(e.id)
                if ei:
                    ei.refresh_geometry(self._node_items.get(e.src_id), self._node_items.get(e.dst_id))

    def _on_output_drag_started(self, node_id: str, start_scene: QPointF) -> None:
        self._clear_temp_wire()
        self._wire_src_id = node_id
        self._temp_wire = QGraphicsLineItem(start_scene.x(), start_scene.y(), start_scene.x(), start_scene.y())
        self._temp_wire.setPen(QPen(QColor(0, 140, 220), 2, Qt.PenStyle.DashLine))
        self._temp_wire.setZValue(1000)
        self._scene.addItem(self._temp_wire)

    def _on_output_drag_moved(self, scene_pos: QPointF) -> None:
        if self._temp_wire is None or self._wire_src_id is None:
            return
        src = self._node_items[self._wire_src_id].output_port_scene_center()
        self._temp_wire.setLine(src.x(), src.y(), scene_pos.x(), scene_pos.y())

    def _on_output_drag_finished(self, scene_pos: QPointF) -> None:
        if self._temp_wire is None or self._wire_src_id is None:
            self._clear_temp_wire()
            return
        src_id = self._wire_src_id
        self._clear_temp_wire()

        dst_id: str | None = None
        for hit in self._scene.items(scene_pos):
            if not isinstance(hit, NodeItem):
                continue
            if hit.port_hit_scene(scene_pos) == "in" and hit.node_id != src_id:
                dst_id = hit.node_id
                break

        if dst_id is None:
            self.status_message.emit(ZH.CANVAS_WIRE_TO_IN_PORT)
            return

        edge, err = self._doc.add_edge(src_id, dst_id)
        if err:
            self.status_message.emit(err)
            return

        ei = EdgeItem(edge.id, edge.src_id, edge.dst_id)
        ei.refresh_geometry(self._node_items.get(edge.src_id), self._node_items.get(edge.dst_id))
        self._scene.addItem(ei)
        self._edge_items[edge.id] = ei
        self.document_modified.emit()
        self.status_message.emit(ZH.CANVAS_WIRE_ADDED)

    def _on_scene_selection_changed(self) -> None:
        sel = self._scene.selectedItems()
        node = None
        for it in sel:
            if isinstance(it, NodeItem):
                node = self._doc.get_node(it.node_id)
                break
        self.selection_node_changed.emit(node)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete:
            self._delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    def _delete_selected(self) -> None:
        sel = list(self._scene.selectedItems())
        if not sel:
            return
        edges = [it for it in sel if isinstance(it, EdgeItem)]
        nodes = [it for it in sel if isinstance(it, NodeItem)]
        for it in edges:
            self._doc.remove_edge(it.edge_id)
            self._scene.removeItem(it)
            self._edge_items.pop(it.edge_id, None)
        for it in nodes:
            nid = it.node_id
            self._doc.remove_node(nid)
            for eid, ei in list(self._edge_items.items()):
                if self._doc.edges.get(eid) is None:
                    self._scene.removeItem(ei)
                    del self._edge_items[eid]
            self._scene.removeItem(it)
            self._node_items.pop(nid, None)
        self.document_modified.emit()
        self._on_scene_selection_changed()

    def add_node_type_at_scene(self, node_type: str, scene_pos: QPointF) -> None:
        """在场景坐标创建节点（左上角对齐放置点）。"""
        x = scene_pos.x() - NODE_WIDTH / 2
        y = scene_pos.y() - NODE_HEIGHT / 2
        gn = self._doc.add_node(node_type, x=x, y=y)
        item = NodeItem(gn.id, palette_label_zh(gn.type))
        item.setPos(QPointF(gn.x, gn.y))
        item.moved.connect(self._on_node_moved)
        item.output_drag_started.connect(self._on_output_drag_started)
        item.output_drag_moved.connect(self._on_output_drag_moved)
        item.output_drag_finished.connect(self._on_output_drag_finished)
        self._scene.addItem(item)
        self._node_items[gn.id] = item
        self.document_modified.emit()

    def duplicate_selected_nodes(self) -> None:
        """复制选中节点（略偏移，不复制连线）。"""
        nodes = [it for it in self._scene.selectedItems() if isinstance(it, NodeItem)]
        if not nodes:
            return
        self._scene.clearSelection()
        for it in nodes:
            gn = self._doc.get_node(it.node_id)
            if gn is None:
                continue
            new_gn = self._doc.add_node(gn.type, x=gn.x + 24, y=gn.y + 24, params=dict(gn.params))
            item = NodeItem(new_gn.id, palette_label_zh(new_gn.type))
            item.setPos(QPointF(new_gn.x, new_gn.y))
            item.moved.connect(self._on_node_moved)
            item.output_drag_started.connect(self._on_output_drag_started)
            item.output_drag_moved.connect(self._on_output_drag_moved)
            item.output_drag_finished.connect(self._on_output_drag_finished)
            self._scene.addItem(item)
            self._node_items[new_gn.id] = item
            item.setSelected(True)
        self.document_modified.emit()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasFormat(MIME_NODE_TYPE):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME_NODE_TYPE):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasFormat(MIME_NODE_TYPE):
            super().dropEvent(event)
            return
        raw = bytes(event.mimeData().data(MIME_NODE_TYPE)).decode("utf-8")
        pos = self.mapToScene(event.position().toPoint())
        self.add_node_type_at_scene(raw.strip(), pos)
        event.acceptProposedAction()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        painter.fillRect(rect, QColor(252, 252, 254))
        grid = 20
        pen = QPen(QColor(230, 232, 238))
        painter.setPen(pen)
        left = int(rect.left()) - (int(rect.left()) % grid)
        top = int(rect.top()) - (int(rect.top()) % grid)
        x = float(left)
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += grid
        y = float(top)
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += grid

    def apply_params_to_selected_node(self, params: dict) -> None:
        sel = [it for it in self._scene.selectedItems() if isinstance(it, NodeItem)]
        if len(sel) != 1:
            return
        nid = sel[0].node_id
        self._doc.update_node_params(nid, params)
        self.document_modified.emit()
