"""主窗口：多 Tab、调色板、参数 Dock、文件与导出菜单。"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
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

from dl_vis.logic.export_torch import export_stub_message
from dl_vis.logic.shape_inference import infer_shapes_linear_nchw
from dl_vis.model.graph_document import GraphDocument, GraphNode
from dl_vis.model.node_types import EDITABLE_FIELDS, param_label_zh
from dl_vis.ui.canvas_widget import CanvasWidget
from dl_vis.ui import locale_zh as ZH
from dl_vis.ui.node_item import NodeItem
from dl_vis.ui.palette import PaletteList


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(ZH.APP_TITLE_BASE)
        self.resize(1280, 780)
        self._dirty = False

        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(self._tabs)
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
        w = self._tabs.currentWidget()
        return w if isinstance(w, CanvasWidget) else None

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

        tool_menu = mb.addMenu(ZH.MENU_TOOL)
        act_shape = QAction(ZH.ACTION_SHAPE_INFER, self)
        act_shape.triggered.connect(self._run_shape_inference)
        tool_menu.addAction(act_shape)

        export_menu = mb.addMenu(ZH.MENU_EXPORT)
        act_export = QAction(ZH.ACTION_EXPORT_TORCH, self)
        act_export.triggered.connect(self._export_torch_stub)
        export_menu.addAction(act_export)

    def _update_title(self) -> None:
        base = ZH.APP_TITLE_BASE
        self.setWindowTitle(f"{base} *" if self._dirty else base)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_title()

    def _mark_clean(self) -> None:
        self._dirty = False
        self._update_title()

    def _new_tab(self, title: str | None = None) -> CanvasWidget:
        canvas = CanvasWidget()
        canvas.selection_node_changed.connect(self._on_selection_node_changed)
        canvas.document_modified.connect(self._mark_dirty)
        canvas.status_message.connect(self._status.setText)
        name = title or ZH.TAB_CANVAS_N.format(self._tabs.count() + 1)
        self._tabs.addTab(canvas, name)
        self._tabs.setCurrentWidget(canvas)
        self._clear_props_form()
        self._mark_dirty()
        return canvas

    def _close_tab(self, index: int) -> None:
        if self._tabs.count() <= 1:
            QMessageBox.information(self, ZH.MSG_HINT_TITLE, ZH.MSG_KEEP_ONE_TAB)
            return
        self._tabs.removeTab(index)
        self._refresh_props_from_canvas()

    def _on_tab_changed(self, _index: int) -> None:
        self._refresh_props_from_canvas()

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
                w.setValue(int(params.get(key, 0)))
                w.valueChanged.connect(lambda v, k=key: self._on_prop_int_changed(k, v))
                self._props_layout.addRow(label, w)
            elif kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(6)
                w.setRange(-1e9, 1e9)
                w.setValue(float(params.get(key, 0.0)))
                w.valueChanged.connect(lambda v, k=key: self._on_prop_float_changed(k, v))
                self._props_layout.addRow(label, w)
            elif kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(params.get(key, False)))
                w.toggled.connect(lambda v, k=key: self._on_prop_bool_changed(k, v))
                self._props_layout.addRow(label, w)
            elif kind == "str":
                w = QLineEdit(str(params.get(key, "")))
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
            return
        doc = c.graph_document()
        n = doc.get_node(self._props_node_id)
        if n is None:
            return
        doc.update_node_params(self._props_node_id, patch)
        self._mark_dirty()

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
        self._tabs.setTabText(self._tabs.indexOf(c), Path(path).name)
        self._mark_clean()
        self._refresh_props_from_canvas()

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
        self._tabs.setTabText(self._tabs.indexOf(c), Path(path).name)
        self._mark_clean()

    def _export_torch_stub(self) -> None:
        QMessageBox.information(self, ZH.DLG_EXPORT_TORCH, export_stub_message())

    def _run_shape_inference(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        res = infer_shapes_linear_nchw(c.graph_document())
        lines = [res.message]
        if res.ok and res.shapes_by_node:
            lines.append("")
            for nid, shp in res.shapes_by_node.items():
                n = c.graph_document().get_node(nid)
                tag = f"{n.type}（ID {nid[:8]}…）" if n else nid
                lines.append(ZH.SHAPE_ROW_PREFIX.format(tag, shp))
        QMessageBox.information(self, ZH.DLG_SHAPE_INFER, "\n".join(lines))
