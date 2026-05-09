"""单页画布：场景、网格、节点/边同步、连线拖拽、删除、撤销重做、对齐。"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QDragEnterEvent, QDropEvent, QKeyEvent, QMouseEvent, QPainter, QPen, QUndoStack
from PyQt6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
)

from dl_vis.model.graph_document import GraphDocument
from dl_vis.model.node_types import palette_label_zh
from dl_vis.ui.edge_item import EdgeItem
from dl_vis.ui.graph_undo import DocStateCommand
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
    undo_state_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc = GraphDocument()
        self._node_items: dict[str, NodeItem] = {}
        self._edge_items: dict[str, EdgeItem] = {}
        self._temp_wire: QGraphicsLineItem | None = None
        self._wire_src_id: str | None = None

        self._undo_stack = QUndoStack(self)
        self._undo_stack.canUndoChanged.connect(self.undo_state_changed.emit)
        self._undo_stack.canRedoChanged.connect(self.undo_state_changed.emit)

        self._pending_drag_undo_snapshot: dict | None = None

        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(QRectF(-2000, -2000, 4000, 4000))
        self.setScene(self._scene)
        # 默认 AnchorViewCenter 会在矩阵刷新时把锚点对准视图中心，拖动节点时易产生「瞬移到中间」错觉；改用 NoAnchor + 左上对齐。
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # RubberBandDrag 与节点拖动易抢占手势；框选用 Ctrl+点击多选代替。
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setAcceptDrops(True)
        self.scale(1.0, 1.0)

        self.viewport().installEventFilter(self)

        self._scene.selectionChanged.connect(self._on_scene_selection_changed)

    def _schedule_finalize_drag_undo(self, *_args: object) -> None:
        """推迟到事件循环下一轮再提交撤销点，确保 Qt 已完成图元位移。"""
        QTimer.singleShot(0, self._finalize_drag_undo_if_needed)

    def undo_stack(self) -> QUndoStack:
        return self._undo_stack

    def push_undo_if_changed(self, before: dict, after: dict, text: str) -> None:
        if json.dumps(before, sort_keys=True, default=str) == json.dumps(after, sort_keys=True, default=str):
            return
        self._undo_stack.push(DocStateCommand(self, before, after, text))

    @staticmethod
    def _same_topology(a: GraphDocument, b: GraphDocument) -> bool:
        """节点/边拓扑与几何一致（仅 params 等可不同）时可跳过 scene.clear，避免在参数控件信号栈里销毁 SpinBox 导致崩溃。"""
        if set(a.nodes.keys()) != set(b.nodes.keys()):
            return False
        if set(a.edges.keys()) != set(b.edges.keys()):
            return False
        for nid, na in a.nodes.items():
            nb = b.nodes[nid]
            if na.type != nb.type or na.x != nb.x or na.y != nb.y:
                return False
        for eid, ea in a.edges.items():
            eb = b.edges[eid]
            if ea.src_id != eb.src_id or ea.dst_id != eb.dst_id:
                return False
        return True

    def apply_doc_snapshot(self, data: dict) -> None:
        """供 DocStateCommand 调用；不清空 QUndoStack。"""
        self._pending_drag_undo_snapshot = None
        new_doc = GraphDocument.from_dict(copy.deepcopy(data))
        if self._same_topology(self._doc, new_doc):
            self._doc = new_doc
            self.document_modified.emit()
            return
        self._doc = new_doc
        self._rebuild_from_document()
        self.document_modified.emit()

    def graph_document(self) -> GraphDocument:
        return self._doc

    def set_graph_document(self, doc: GraphDocument, *, clear_undo: bool = True) -> None:
        if clear_undo:
            self._undo_stack.clear()
        self._pending_drag_undo_snapshot = None
        self._doc = doc
        self._rebuild_from_document()

    def _wire_node_item(self, item: NodeItem) -> None:
        item.position_changed.connect(self._on_node_position_changed_live)
        item.live_drag_finished.connect(self._schedule_finalize_drag_undo)
        item.output_drag_started.connect(self._on_output_drag_started)
        item.output_drag_moved.connect(self._on_output_drag_moved)
        item.output_drag_finished.connect(self._on_output_drag_finished)

    def _rebuild_from_document(self) -> None:
        self._scene.blockSignals(True)
        self._scene.clear()
        self._node_items.clear()
        self._edge_items.clear()
        self._clear_temp_wire()

        for gn in self._doc.iter_nodes():
            item = NodeItem(gn.id, palette_label_zh(gn.type))
            item.setPos(QPointF(gn.x, gn.y))
            self._wire_node_item(item)
            self._scene.addItem(item)
            self._node_items[gn.id] = item

        for ge in self._doc.iter_edges():
            ei = EdgeItem(ge.id, ge.src_id, ge.dst_id)
            ei.refresh_geometry(self._node_items.get(ge.src_id), self._node_items.get(ge.dst_id))
            self._scene.addItem(ei)
            self._edge_items[ge.id] = ei

        self._scene.blockSignals(False)
        self._on_scene_selection_changed()

    def eventFilter(self, obj, event: QEvent) -> bool:
        if obj is self.viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            me = event
            if isinstance(me, QMouseEvent) and me.button() == Qt.MouseButton.LeftButton:
                self._schedule_finalize_drag_undo()
        return super().eventFilter(obj, event)

    def _finalize_drag_undo_if_needed(self) -> None:
        if self._pending_drag_undo_snapshot is None:
            return
        before = self._pending_drag_undo_snapshot
        self._pending_drag_undo_snapshot = None
        after = copy.deepcopy(self._doc.to_dict())
        if json.dumps(before, sort_keys=True, default=str) == json.dumps(after, sort_keys=True, default=str):
            return
        self.push_undo_if_changed(before, after, ZH.UNDO_MOVE_NODE)
        self.document_modified.emit()

    def _clear_temp_wire(self) -> None:
        if self._temp_wire is not None:
            self._scene.removeItem(self._temp_wire)
            self._temp_wire = None
        self._wire_src_id = None

    def _on_node_position_changed_live(self, node_id: str) -> None:
        """拖动过程中持续把场景坐标写回 GraphDocument，避免模型与视图不一致导致瞬移。"""
        if self._pending_drag_undo_snapshot is None:
            self._pending_drag_undo_snapshot = copy.deepcopy(self._doc.to_dict())
        item = self._node_items.get(node_id)
        if item is not None:
            sp = item.scenePos()
            self._doc.update_node_position(node_id, sp.x(), sp.y())
        self._refresh_edges_for(node_id)

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

        before = copy.deepcopy(self._doc.to_dict())
        edge, err = self._doc.add_edge(src_id, dst_id)
        if err:
            self.status_message.emit(err)
            return

        ei = EdgeItem(edge.id, edge.src_id, edge.dst_id)
        ei.refresh_geometry(self._node_items.get(edge.src_id), self._node_items.get(edge.dst_id))
        self._scene.addItem(ei)
        self._edge_items[edge.id] = ei
        after = copy.deepcopy(self._doc.to_dict())
        self.push_undo_if_changed(before, after, ZH.UNDO_ADD_EDGE)
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
        before = copy.deepcopy(self._doc.to_dict())
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
        after = copy.deepcopy(self._doc.to_dict())
        self.push_undo_if_changed(before, after, ZH.UNDO_DELETE)
        self.document_modified.emit()
        self._on_scene_selection_changed()

    def add_node_type_at_scene(self, node_type: str, scene_pos: QPointF) -> None:
        """在场景坐标创建节点（左上角对齐放置点）。"""
        before = copy.deepcopy(self._doc.to_dict())
        x = scene_pos.x() - NODE_WIDTH / 2
        y = scene_pos.y() - NODE_HEIGHT / 2
        gn = self._doc.add_node(node_type, x=x, y=y)
        item = NodeItem(gn.id, palette_label_zh(gn.type))
        item.setPos(QPointF(gn.x, gn.y))
        self._wire_node_item(item)
        self._scene.addItem(item)
        self._node_items[gn.id] = item
        after = copy.deepcopy(self._doc.to_dict())
        self.push_undo_if_changed(before, after, ZH.UNDO_ADD_NODE)
        self.document_modified.emit()

    def duplicate_selected_nodes(self) -> None:
        """复制选中节点（略偏移，不复制连线）。"""
        nodes = [it for it in self._scene.selectedItems() if isinstance(it, NodeItem)]
        if not nodes:
            return
        before = copy.deepcopy(self._doc.to_dict())
        self._scene.clearSelection()
        for it in nodes:
            gn = self._doc.get_node(it.node_id)
            if gn is None:
                continue
            new_gn = self._doc.add_node(gn.type, x=gn.x + 24, y=gn.y + 24, params=dict(gn.params))
            item = NodeItem(new_gn.id, palette_label_zh(new_gn.type))
            item.setPos(QPointF(new_gn.x, new_gn.y))
            self._wire_node_item(item)
            self._scene.addItem(item)
            self._node_items[new_gn.id] = item
            item.setSelected(True)
        after = copy.deepcopy(self._doc.to_dict())
        self.push_undo_if_changed(before, after, ZH.UNDO_DUPLICATE)
        self.document_modified.emit()

    def align_selected_nodes(self, mode: str) -> None:
        """mode: 'left' | 'top'"""
        nodes = [it for it in self._scene.selectedItems() if isinstance(it, NodeItem)]
        if len(nodes) < 2:
            self.status_message.emit(ZH.ALIGN_NEED_TWO)
            return
        before = copy.deepcopy(self._doc.to_dict())
        if mode == "left":
            target_x = min(it.pos().x() for it in nodes)
            for it in nodes:
                it.setPos(target_x, it.pos().y())
        elif mode == "top":
            target_y = min(it.pos().y() for it in nodes)
            for it in nodes:
                it.setPos(it.pos().x(), target_y)
        else:
            return
        for nid, item in self._node_items.items():
            self._doc.update_node_position(nid, item.pos().x(), item.pos().y())
        for it in nodes:
            self._refresh_edges_for(it.node_id)
        after = copy.deepcopy(self._doc.to_dict())
        self.push_undo_if_changed(before, after, ZH.UNDO_ALIGN)
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
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
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

    def push_param_undo(self, before: dict, after: dict) -> None:
        self.push_undo_if_changed(before, after, ZH.UNDO_EDIT_PARAM)
