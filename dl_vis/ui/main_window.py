"""主窗口：多 Tab 画布、调色板、参数 Dock、文件与导出菜单。"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dl_vis.logic.export_torch import export_sequential_source, export_stub_message
from dl_vis.logic.shape_inference import infer_shapes_linear_nchw
from dl_vis.model.graph_document import GraphDocument, GraphNode
from dl_vis.model.node_types import EDITABLE_FIELDS, param_label_zh
from dl_vis.ui.canvas_widget import CanvasWidget
from dl_vis.ui import locale_zh as ZH
from dl_vis.ui.node_item import NodeItem
from dl_vis.ui.palette import PaletteList
from dl_vis.ui.viz_tab import VizTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(ZH.APP_TITLE_BASE)
        self.resize(1280, 780)
        self._dirty = False

        self._canvas_tabs = QTabWidget()
        self._canvas_tabs.setTabsClosable(True)
        self._canvas_tabs.tabCloseRequested.connect(self._close_tab)
        self._canvas_tabs.currentChanged.connect(self._on_canvas_tab_changed)

        self._viz_tab = VizTab(self)

        self._central_tabs = QTabWidget()
        self._central_tabs.addTab(self._canvas_tabs, ZH.TAB_EDITOR)
        self._central_tabs.addTab(self._viz_tab, ZH.TAB_VIZ)
        self._central_tabs.currentChanged.connect(self._on_central_tab_changed)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(self._central_tabs)
        self.setCentralWidget(central)

        self._palette_dock = QDockWidget(ZH.DOCK_PALETTE, self)
        self._palette = PaletteList()
        self._palette_dock.setWidget(self._palette)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._palette_dock)

        self._props_container = QWidget()
        self._props_layout = QFormLayout(self._props_container)
        self._props_scroll = QScrollArea()
        self._props_scroll.setWidgetResizable(True)
        self._props_scroll.setWidget(self._props_container)

        self._props_dock = QDockWidget(ZH.DOCK_PARAMS, self)
        self._props_dock.setWidget(self._props_scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._props_dock)

        self._status = QLabel("")
        self.statusBar().addWidget(self._status, 1)

        self._props_node_id: str | None = None
        self._props_loading = False

        self._build_menu()
        self._new_tab(ZH.TAB_CANVAS_N.format(1))
        self._mark_clean()

    def _current_canvas(self) -> CanvasWidget | None:
        w = self._canvas_tabs.currentWidget()
        return w if isinstance(w, CanvasWidget) else None

    def _schedule_viz_refresh(self) -> None:
        """勿在参数控件信号栈内直接 draw matplotlib；推迟且仅在「可视化」页显示时刷新。"""
        QTimer.singleShot(0, self._flush_viz_refresh_safe)

    def _flush_viz_refresh_safe(self) -> None:
        if self._central_tabs.currentWidget() is not self._viz_tab:
            return
        try:
            self._viz_tab.refresh_from_main()
        except Exception:
            logging.getLogger("dl_vis").exception("可视化 Tab 刷新失败")
            self._status.setText("可视化刷新失败，详见控制台/stderr 日志。")

    def _build_menu(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu(ZH.MENU_FILE)

        act_new_page = QAction(ZH.ACTION_NEW_TAB, self)
        act_new_page.setShortcut(QKeySequence.StandardKey.New)
        act_new_page.triggered.connect(lambda: self._new_tab())
        file_menu.addAction(act_new_page)

        act_open = QAction(ZH.ACTION_OPEN, self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_graph)
        file_menu.addAction(act_open)

        act_save = QAction(ZH.ACTION_SAVE, self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_graph)
        file_menu.addAction(act_save)

        act_dup = QAction(ZH.ACTION_COPY_NODES, self)
        act_dup.setShortcut(QKeySequence("Ctrl+D"))
        act_dup.triggered.connect(self._duplicate_selected)
        file_menu.addAction(act_dup)

        edit_menu = mb.addMenu(ZH.MENU_EDIT)

        self._act_undo = QAction(ZH.ACTION_UNDO, self)
        self._act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._act_undo.triggered.connect(self._do_undo)
        self._act_undo.setEnabled(False)
        edit_menu.addAction(self._act_undo)

        self._act_redo = QAction(ZH.ACTION_REDO, self)
        self._act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self._act_redo.triggered.connect(self._do_redo)
        self._act_redo.setEnabled(False)
        edit_menu.addAction(self._act_redo)

        edit_menu.addSeparator()

        act_align_left = QAction(ZH.ACTION_ALIGN_LEFT, self)
        act_align_left.triggered.connect(lambda: self._align_nodes("left"))
        edit_menu.addAction(act_align_left)

        act_align_top = QAction(ZH.ACTION_ALIGN_TOP, self)
        act_align_top.triggered.connect(lambda: self._align_nodes("top"))
        edit_menu.addAction(act_align_top)

        tool_menu = mb.addMenu(ZH.MENU_TOOL)
        act_shape = QAction(ZH.ACTION_SHAPE_INFER, self)
        act_shape.triggered.connect(self._run_shape_inference)
        tool_menu.addAction(act_shape)

        export_menu = mb.addMenu(ZH.MENU_EXPORT)

        act_export_help = QAction(ZH.ACTION_EXPORT_TORCH, self)
        act_export_help.triggered.connect(self._export_torch_help)
        export_menu.addAction(act_export_help)

        act_export_copy = QAction(ZH.ACTION_EXPORT_COPY, self)
        act_export_copy.triggered.connect(self._export_copy_sequential)
        export_menu.addAction(act_export_copy)

        act_export_save = QAction(ZH.ACTION_EXPORT_SAVE, self)
        act_export_save.triggered.connect(self._export_save_sequential)
        export_menu.addAction(act_export_save)

    def _do_undo(self) -> None:
        c = self._current_canvas()
        if c:
            c.undo_stack().undo()

    def _do_redo(self) -> None:
        c = self._current_canvas()
        if c:
            c.undo_stack().redo()

    def _sync_undo_actions(self) -> None:
        c = self._current_canvas()
        stack = c.undo_stack() if c else None
        self._act_undo.setEnabled(stack.canUndo() if stack else False)
        self._act_redo.setEnabled(stack.canRedo() if stack else False)

    def _align_nodes(self, mode: str) -> None:
        c = self._current_canvas()
        if c:
            c.align_selected_nodes(mode)

    def _export_torch_help(self) -> None:
        QMessageBox.information(self, ZH.DLG_EXPORT_TORCH, export_stub_message())

    def _export_copy_sequential(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        try:
            src = export_sequential_source(c.graph_document())
        except ValueError as e:
            QMessageBox.warning(self, ZH.EXPORT_FAIL_TITLE, str(e))
            return
        QApplication.clipboard().setText(src)
        self._status.setText("已复制 Sequential 源码到剪贴板。")

    def _export_save_sequential(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        try:
            src = export_sequential_source(c.graph_document())
        except ValueError as e:
            QMessageBox.warning(self, ZH.EXPORT_FAIL_TITLE, str(e))
            return
        path, _ = QFileDialog.getSaveFileName(self, ZH.ACTION_EXPORT_SAVE, "", "Python (*.py)")
        if not path:
            return
        if not path.endswith(".py"):
            path += ".py"
        try:
            Path(path).write_text(src, encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, ZH.MSG_SAVE_FAIL, str(e))
            return
        self._status.setText(f"已保存：{path}")

    def _update_title(self) -> None:
        base = ZH.APP_TITLE_BASE
        self.setWindowTitle(f"{base} *" if self._dirty else base)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_title()

    def _mark_clean(self) -> None:
        self._dirty = False
        self._update_title()

    def _wire_canvas(self, canvas: CanvasWidget) -> None:
        canvas.selection_node_changed.connect(self._on_selection_node_changed)
        canvas.document_modified.connect(self._mark_dirty)
        canvas.status_message.connect(self._status.setText)
        canvas.undo_stack().canUndoChanged.connect(self._sync_undo_actions)
        canvas.undo_stack().canRedoChanged.connect(self._sync_undo_actions)

    def _new_tab(self, title: str | None = None) -> CanvasWidget:
        canvas = CanvasWidget()
        self._wire_canvas(canvas)
        name = title or ZH.TAB_CANVAS_N.format(self._canvas_tabs.count() + 1)
        self._canvas_tabs.addTab(canvas, name)
        self._canvas_tabs.setCurrentWidget(canvas)
        self._clear_props_form()
        self._mark_dirty()
        self._sync_undo_actions()
        return canvas

    def _close_tab(self, index: int) -> None:
        if self._canvas_tabs.count() <= 1:
            QMessageBox.information(self, ZH.MSG_HINT_TITLE, ZH.MSG_KEEP_ONE_TAB)
            return
        self._canvas_tabs.removeTab(index)
        self._refresh_props_from_canvas()
        self._sync_undo_actions()
        self._schedule_viz_refresh()

    def _on_canvas_tab_changed(self, _index: int) -> None:
        self._refresh_props_from_canvas()
        self._sync_undo_actions()
        self._schedule_viz_refresh()

    def _on_central_tab_changed(self, _index: int) -> None:
        self._schedule_viz_refresh()

    def _refresh_props_from_canvas(self) -> None:
        c = self._current_canvas()
        if c is None:
            self._clear_props_form()
            return
        node = None
        for it in c.scene().selectedItems():
            if isinstance(it, NodeItem):
                node = c.graph_document().get_node(it.node_id)
                break
        self._fill_props_form(node)

    def _on_selection_node_changed(self, node: object) -> None:
        self._fill_props_form(node if isinstance(node, GraphNode) else None)
        self._schedule_viz_refresh()

    def _clear_props_form(self) -> None:
        while self._props_layout.rowCount():
            self._props_layout.removeRow(0)
        self._props_node_id = None

    def _fill_props_form(self, node: GraphNode | None) -> None:
        self._clear_props_form()
        if node is None:
            self._props_layout.addRow(QLabel(ZH.PROP_SELECT_NODE))
            return
        self._props_node_id = node.id
        fields = EDITABLE_FIELDS.get(node.type, [])
        if not fields:
            self._props_layout.addRow(QLabel(ZH.PROP_NO_FIELDS))
            return

        self._props_loading = True
        params = node.params
        for key, kind in fields:
            label = param_label_zh(key)
            if kind == "int":
                w = QSpinBox()
                w.setRange(-999999999, 999999999)
                w.blockSignals(True)
                w.setValue(int(params.get(key, 0)))
                w.blockSignals(False)
                w.valueChanged.connect(lambda v, k=key: self._on_prop_int_changed(k, v))
                self._props_layout.addRow(label, w)
            elif kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(6)
                w.setRange(-1e9, 1e9)
                w.blockSignals(True)
                w.setValue(float(params.get(key, 0.0)))
                w.blockSignals(False)
                w.valueChanged.connect(lambda v, k=key: self._on_prop_float_changed(k, v))
                self._props_layout.addRow(label, w)
            elif kind == "bool":
                w = QCheckBox()
                w.blockSignals(True)
                w.setChecked(bool(params.get(key, False)))
                w.blockSignals(False)
                w.toggled.connect(lambda v, k=key: self._on_prop_bool_changed(k, v))
                self._props_layout.addRow(label, w)
            elif kind == "str":
                w = QLineEdit()
                w.blockSignals(True)
                w.setText(str(params.get(key, "")))
                w.blockSignals(False)
                w.editingFinished.connect(lambda k=key, widget=w: self._on_prop_str_changed(k, widget.text()))
                self._props_layout.addRow(label, w)
        self._props_loading = False

    def _on_prop_int_changed(self, key: str, value: int) -> None:
        if self._props_loading:
            return
        self._apply_param({key: value})

    def _on_prop_float_changed(self, key: str, value: float) -> None:
        if self._props_loading:
            return
        self._apply_param({key: value})

    def _on_prop_bool_changed(self, key: str, value: bool) -> None:
        if self._props_loading:
            return
        self._apply_param({key: value})

    def _on_prop_str_changed(self, key: str, text: str) -> None:
        if self._props_loading:
            return
        self._apply_param({key: text})

    def _apply_param(self, patch: dict) -> None:
        c = self._current_canvas()
        if c is None or self._props_node_id is None:
            logging.getLogger("dl_vis.ui").debug("_apply_param：无画布或未选中节点，忽略 patch=%s", patch)
            return
        doc = c.graph_document()
        n = doc.get_node(self._props_node_id)
        if n is None:
            logging.getLogger("dl_vis.ui").warning("_apply_param：节点 id=%s 已不存在", self._props_node_id)
            return
        log = logging.getLogger("dl_vis.ui")
        log.debug("参数变更 node=%s type=%s patch=%s", self._props_node_id[:8], n.type, patch)
        before = copy.deepcopy(doc.to_dict())
        doc.update_node_params(self._props_node_id, patch)
        after = copy.deepcopy(doc.to_dict())
        try:
            c.push_param_undo(before, after)
        except Exception:
            log.exception("提交参数撤销点失败（patch=%s）", patch)
            self._status.setText("参数保存失败，详见日志。")
            return
        self._mark_dirty()
        self._schedule_viz_refresh()

    def _duplicate_selected(self) -> None:
        c = self._current_canvas()
        if c:
            c.duplicate_selected_nodes()

    def _open_graph(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, ZH.DLG_OPEN_GRAPH, "", f"{ZH.FILE_FILTER_JSON};;{ZH.FILE_FILTER_ALL}"
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
            doc = GraphDocument.from_json(text)
        except OSError as e:
            QMessageBox.warning(self, ZH.MSG_OPEN_FAIL, str(e))
            return
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            QMessageBox.warning(self, ZH.MSG_PARSE_FAIL, str(e))
            return
        c = self._current_canvas()
        if c is None:
            c = self._new_tab()
        c.set_graph_document(doc)
        self._canvas_tabs.setTabText(self._canvas_tabs.indexOf(c), Path(path).name)
        self._mark_clean()
        self._refresh_props_from_canvas()
        self._sync_undo_actions()
        self._schedule_viz_refresh()

    def _save_graph(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, ZH.DLG_SAVE_GRAPH, "", f"{ZH.FILE_FILTER_JSON};;{ZH.FILE_FILTER_ALL}"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            Path(path).write_text(c.graph_document().to_json(), encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, ZH.MSG_SAVE_FAIL, str(e))
            return
        self._canvas_tabs.setTabText(self._canvas_tabs.indexOf(c), Path(path).name)
        self._mark_clean()

    def _run_shape_inference(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        res = infer_shapes_linear_nchw(c.graph_document())
        lines = [res.message, "", ZH.SHAPE_SUPPORT_NOTE]
        if res.warnings:
            lines.extend(["", "警告："])
            lines.extend(res.warnings)
        if res.ok and res.shapes_by_node:
            lines.append("")
            for nid, shp in res.shapes_by_node.items():
                n = c.graph_document().get_node(nid)
                tag = f"{n.type}（ID {nid[:8]}…）" if n else nid
                lines.append(ZH.SHAPE_ROW_PREFIX.format(tag, shp))
        QMessageBox.information(self, ZH.DLG_SHAPE_INFER, "\n".join(lines))
