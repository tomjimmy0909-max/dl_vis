"""主窗口：多 Tab 画布、调色板、参数 Dock、文件与导出菜单。"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QCursor, QFont, QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dl_vis.logic.export_torch import export_full_training_script, export_stub_message
from dl_vis.logic.model_ast_import import graph_document_from_py_file
from dl_vis.logic.runtime_torch import (
    MSG_TORCH_MISSING,
    check_graph_runnable,
    dummy_forward,
    validate_output_for_train,
)
from dl_vis.logic.node_code_apply import apply_code_preview_text, code_params_editable
from dl_vis.logic.node_code_preview import code_preview_for_node
from dl_vis.logic.shape_inference import export_sequential_prerequisite_message, infer_shapes_dag_nchw
from dl_vis.model.graph_document import GraphDocument, GraphNode
from dl_vis.model.node_types import EDITABLE_FIELDS, param_label_zh
from dl_vis.ui.canvas_widget import CanvasWidget
from dl_vis.ui.data_proc_panel import DataProcPanel
from dl_vis.ui import locale_zh as ZH
from dl_vis.ui.materials_panel import MaterialsPanel
from dl_vis.ui.node_item import NodeItem
from dl_vis.ui.palette import PaletteList
from dl_vis.ui.training_worker import TrainingJobConfig, TrainingWorker
from dl_vis.ui.viz_tab import VizTab


EMPTY_PY_TEMPLATE = '''"""dl_vis 空模板：可在「开始 → 将绑定脚本解析为流程图」中导入为画布。"""
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(16 * 112 * 112, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    m = Net()
    y = m(torch.randn(1, 3, 224, 224))
    print(y.shape)
'''
class MainWindow(QMainWindow):
    """
    dl_vis 主窗口。

    布局：
    - 中央：图形编辑 Tab（画布）+ 可视化 Tab
    - 左侧 Dock：算子调色板 + 训练素材面板
    - 右侧 Dock：参数表单 + 代码预览/编辑

    功能入口：
    - 菜单栏：开始、文件、编辑、视图、工具、运行、导出
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(ZH.APP_TITLE_BASE)
        self.resize(1280, 780)
        self._dirty = False          # 文档是否被修改（标题栏显示 *）
        self._workspace_root: Path | None = None  # 当前工作目录

        # === 中央区域：画布 Tab + 可视化 Tab ===
        self._canvas_tabs = QTabWidget()
        self._canvas_tabs.setTabsClosable(True)
        self._canvas_tabs.tabCloseRequested.connect(self._close_tab)
        self._canvas_tabs.currentChanged.connect(self._on_canvas_tab_changed)

        self._viz_tab = VizTab(self)  # 可视化 Tab（Matplotlib 参数条形图）

        self._central_tabs = QTabWidget()
        self._central_tabs.addTab(self._canvas_tabs, ZH.TAB_EDITOR)
        self._central_tabs.addTab(self._viz_tab, ZH.TAB_VIZ)
        self._central_tabs.currentChanged.connect(self._on_central_tab_changed)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(self._central_tabs)
        self.setCentralWidget(central)

        # === 左侧 Dock：算子调色板 + 训练素材 ===
        self._palette_dock = QDockWidget(ZH.DOCK_PALETTE, self)
        self._palette = PaletteList()
        self._materials_panel = MaterialsPanel()
        self._data_proc_panel = DataProcPanel(self)
        self._left_panel_tabs = QTabWidget()
        self._left_panel_tabs.addTab(self._palette, ZH.TAB_PALETTE_NODES)
        self._left_panel_tabs.addTab(self._materials_panel, ZH.TAB_MATERIALS)
        self._left_panel_tabs.addTab(self._data_proc_panel, ZH.TAB_DATA_PROC)
        self._palette_dock.setWidget(self._left_panel_tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._palette_dock)

        # === 右侧 Dock：参数表单 + 代码预览/编辑 ===
        self._props_container = QWidget()
        props_outer = QVBoxLayout(self._props_container)
        props_outer.setContentsMargins(0, 0, 0, 0)
        form_host = QWidget()
        self._props_layout = QFormLayout(form_host)  # 动态生成的参数表单
        props_outer.addWidget(form_host)
        props_outer.addWidget(QLabel(ZH.PROP_CODE_PREVIEW_HEAD))
        self._code_hint = QLabel(ZH.PROP_CODE_HINT_NONE)
        self._code_hint.setWordWrap(True)
        props_outer.addWidget(self._code_hint)
        self._code_preview = QPlainTextEdit()  # 代码预览/编辑区
        self._code_preview.setReadOnly(True)
        self._code_preview.setPlaceholderText(ZH.PROP_CODE_PREVIEW_PLACEHOLDER)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._code_preview.setFont(mono)
        self._code_preview.setMinimumHeight(140)
        props_outer.addWidget(self._code_preview, 1)
        self._btn_apply_code = QPushButton(ZH.ACTION_APPLY_CODE)  # 「应用代码」按钮
        self._btn_apply_code.setEnabled(False)
        self._btn_apply_code.clicked.connect(self._apply_code_from_preview)
        props_outer.addWidget(self._btn_apply_code)
        # 快捷键 Ctrl+Enter 提交代码
        sc_apply = QShortcut(QKeySequence("Ctrl+Return"), self._code_preview)
        sc_apply.setContext(Qt.ShortcutContext.WidgetShortcut)
        sc_apply.activated.connect(self._apply_code_from_preview)
        self._props_scroll = QScrollArea()
        self._props_scroll.setWidgetResizable(True)
        self._props_scroll.setWidget(self._props_container)

        self._props_dock = QDockWidget(ZH.DOCK_PARAMS, self)
        self._props_dock.setWidget(self._props_scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._props_dock)

        # Dock 特性：可移动、可浮动、可关闭
        _dock_feats = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self._palette_dock.setFeatures(_dock_feats)
        self._props_dock.setFeatures(_dock_feats)

        # 状态栏
        self._status = QLabel("")
        self.statusBar().addWidget(self._status, 1)

        # 参数面板状态
        self._props_node_id: str | None = None  # 当前正在编辑的节点 ID
        self._props_loading = False              # 参数面板正在加载中（避免信号循环）
        self._train_worker: TrainingWorker | None = None  # 后台训练线程

        # 构建菜单栏并创建第一个画布 Tab
        self._build_menu()
        self._new_tab(ZH.TAB_CANVAS_N.format(1))
        self._mark_clean()
        self._update_start_actions_enabled()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # 部分终端/多显示器 + DPI 下，首次 show 立刻 move 时 frameGeometry 不稳定，易把标题栏移出工作区
        # （看起来整屏的内容、看不到最大化/关闭）。延后一帧再居中并夹在工作区内。
        if getattr(self, "_dl_vis_did_center_window", False):
            return
        self._dl_vis_did_center_window = True
        QTimer.singleShot(0, self._place_window_in_work_area)

    def _place_window_in_work_area(self) -> None:
        center_hint = self.frameGeometry().center()
        screen = (
            QGuiApplication.screenAt(center_hint)
            or QGuiApplication.screenAt(QCursor.pos())
            or self.screen()
            or QGuiApplication.primaryScreen()
        )
        if screen is None:
            return
        ag = screen.availableGeometry()
        inner = self.geometry()
        frame = self.frameGeometry()
        dx = frame.width() - inner.width()
        dy = frame.height() - inner.height()
        max_inner_w = max(400, ag.width() - dx)
        max_inner_h = max(300, ag.height() - dy)
        if inner.width() > max_inner_w or inner.height() > max_inner_h:
            self.resize(
                min(inner.width(), max_inner_w),
                min(inner.height(), max_inner_h),
            )
        fg = self.frameGeometry()
        fg.moveCenter(ag.center())
        if fg.left() < ag.left():
            fg.moveLeft(ag.left())
        if fg.top() < ag.top():
            fg.moveTop(ag.top())
        if fg.right() > ag.right():
            fg.moveLeft(ag.right() - fg.width())
        if fg.bottom() > ag.bottom():
            fg.moveTop(ag.bottom() - fg.height())
        self.move(fg.topLeft())

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
        """
        构建完整菜单栏。

        菜单结构：
        - 开始：打开 .py / 文件夹 / 创建模板 / 解析为流程图
        - 文件：新建 Tab / 打开 JSON / 保存 JSON / 复制节点
        - 编辑：撤销 / 重做 / 左对齐 / 顶对齐
        - 视图：重置停靠面板
        - 工具：形状推导
        - 运行：试跑前向 / 合成数据训练
        - 导出：帮助 / 复制 Sequential 源码 / 保存为 .py
        """
        mb = self.menuBar()

        # === 「开始」菜单：与外部文件交互 ===
        start_menu = mb.addMenu(ZH.MENU_START)
        act_open_py = QAction(ZH.ACTION_START_OPEN_PY, self)
        act_open_py.triggered.connect(self._start_open_py_file)
        start_menu.addAction(act_open_py)
        act_open_dir = QAction(ZH.ACTION_START_OPEN_FOLDER, self)
        act_open_dir.triggered.connect(self._start_open_folder)
        start_menu.addAction(act_open_dir)
        act_tpl = QAction(ZH.ACTION_START_NEW_TEMPLATE, self)
        act_tpl.triggered.connect(self._start_create_template)
        start_menu.addAction(act_tpl)
        start_menu.addSeparator()
        self._act_parse_script = QAction(ZH.ACTION_PARSE_SCRIPT, self)
        self._act_parse_script.triggered.connect(self._parse_bound_script_to_graph)
        self._act_parse_script.setEnabled(False)
        start_menu.addAction(self._act_parse_script)

        # === 「文件」菜单：画布文档管理 ===
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

        # === 「编辑」菜单：撤销/重做/对齐 ===
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

        # === 「视图」菜单 ===
        view_menu = mb.addMenu(ZH.MENU_VIEW)
        act_reset_docks = QAction(ZH.ACTION_RESET_DOCKS, self)
        act_reset_docks.triggered.connect(self._reset_dock_layout)
        view_menu.addAction(act_reset_docks)

        # === 「工具」菜单 ===
        tool_menu = mb.addMenu(ZH.MENU_TOOL)
        act_shape = QAction(ZH.ACTION_SHAPE_INFER, self)
        act_shape.triggered.connect(self._run_shape_inference)
        tool_menu.addAction(act_shape)

        # === 「运行」菜单：前向试跑和训练 ===
        run_menu = mb.addMenu(ZH.MENU_RUN)
        act_fwd = QAction(ZH.ACTION_DUMMY_FORWARD, self)
        act_fwd.triggered.connect(self._run_dummy_forward)
        run_menu.addAction(act_fwd)
        act_train = QAction(ZH.ACTION_SYNTHETIC_TRAIN, self)
        act_train.triggered.connect(self._open_train_dialog)
        run_menu.addAction(act_train)

        # === 「导出」菜单：PyTorch 代码生成 ===
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

    def _start_open_py_file(self) -> None:
        start_dir = str(self._workspace_root or Path.home())
        path_str, _ = QFileDialog.getOpenFileName(self, ZH.START_OPEN_PY_TITLE, start_dir, ZH.FILTER_PY)
        if not path_str:
            return
        path = Path(path_str)
        c = self._current_canvas()
        if c is None:
            c = self._new_tab(path.stem)
        else:
            idx = self._canvas_tabs.indexOf(c)
            if idx >= 0:
                self._canvas_tabs.setTabText(idx, path.stem)
        c.set_bound_py_path(path)
        self._update_start_actions_enabled()
        self._status.setText(ZH.MSG_BOUND_SCRIPT.format(str(path)))

    def _start_open_folder(self) -> None:
        start_dir = str(self._workspace_root or Path.home())
        d = QFileDialog.getExistingDirectory(self, ZH.START_OPEN_FOLDER_TITLE, start_dir)
        if not d:
            return
        self._workspace_root = Path(d)
        self._status.setText(ZH.MSG_WORKSPACE_SET.format(d))

    def _start_create_template(self) -> None:
        default_path = (self._workspace_root or Path.home()) / "model_template.py"
        path_str, _ = QFileDialog.getSaveFileName(
            self, ZH.START_SAVE_TEMPLATE_TITLE, str(default_path), ZH.FILTER_PY
        )
        if not path_str:
            return
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(EMPTY_PY_TEMPLATE, encoding="utf-8")
        c = self._current_canvas()
        if c is None:
            c = self._new_tab(path.stem)
        else:
            idx = self._canvas_tabs.indexOf(c)
            if idx >= 0:
                self._canvas_tabs.setTabText(idx, path.stem)
        c.set_bound_py_path(path)
        self._update_start_actions_enabled()
        self._status.setText(ZH.MSG_BOUND_SCRIPT.format(str(path)))
        QMessageBox.information(self, ZH.MSG_HINT_TITLE, f"已写入模板：\n{path}")

    def _parse_bound_script_to_graph(self) -> None:
        c = self._current_canvas()
        if c is None or c.bound_py_path() is None:
            QMessageBox.information(self, ZH.MSG_HINT_TITLE, ZH.MSG_PARSE_NEED_BIND)
            return
        bp = c.bound_py_path()
        assert bp is not None
        try:
            doc = graph_document_from_py_file(bp)
        except ValueError as e:
            QMessageBox.warning(self, ZH.MSG_PARSE_MODEL_TITLE, str(e))
            return
        c.set_graph_document(doc, clear_undo=True)
        self._refresh_props_from_canvas()
        self._mark_dirty()
        self._sync_undo_actions()
        self._schedule_viz_refresh()
        self._status.setText(ZH.MSG_PARSE_OK)

    def _update_start_actions_enabled(self) -> None:
        c = self._current_canvas()
        self._act_parse_script.setEnabled(c is not None and c.bound_py_path() is not None)

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

    def _reset_dock_layout(self) -> None:
        """将左右 Dock 恢复为默认停靠区并取消浮动，避免面板漂在角落无法操作。"""
        self.removeDockWidget(self._palette_dock)
        self.removeDockWidget(self._props_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._palette_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._props_dock)
        self._palette_dock.setFloating(False)
        self._props_dock.setFloating(False)
        self._palette_dock.show()
        self._props_dock.show()
        self._status.setText(ZH.MSG_DOCK_RESET_DONE)

    def _export_torch_help(self) -> None:
        QMessageBox.information(self, ZH.DLG_EXPORT_TORCH, export_stub_message())

    def _export_copy_sequential(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        doc = c.graph_document()
        pre = export_sequential_prerequisite_message(doc)
        if pre:
            QMessageBox.warning(self, ZH.EXPORT_FAIL_TITLE, pre)
            return
        try:
            src = export_full_training_script(doc)
        except ValueError as e:
            QMessageBox.warning(self, ZH.EXPORT_FAIL_TITLE, str(e))
            return
        QApplication.clipboard().setText(src)
        self._status.setText("已复制 Sequential 源码到剪贴板。")

    def _export_save_sequential(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        doc = c.graph_document()
        pre = export_sequential_prerequisite_message(doc)
        if pre:
            QMessageBox.warning(self, ZH.EXPORT_FAIL_TITLE, pre)
            return
        try:
            src = export_full_training_script(doc)
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
        self._update_start_actions_enabled()
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
        self._code_preview.clear()
        self._code_hint.setText(ZH.PROP_CODE_HINT_NONE)
        self._code_preview.setReadOnly(True)
        self._code_preview.setPlaceholderText(ZH.PROP_CODE_PREVIEW_PLACEHOLDER)
        self._btn_apply_code.setEnabled(False)

    def _sync_code_panel(self, node: GraphNode | None) -> None:
        if node is None:
            return
        ok = code_params_editable(node.type)
        self._code_preview.setReadOnly(not ok)
        self._btn_apply_code.setEnabled(ok)
        self._code_hint.setText(ZH.PROP_CODE_HINT_EDITABLE if ok else ZH.PROP_CODE_HINT_VIEWONLY)

    def _fill_props_form(self, node: GraphNode | None) -> None:
        """
        填充右侧参数面板表单。

        根据节点类型从 EDITABLE_FIELDS 获取可编辑字段定义，
        动态生成对应类型的 Qt 控件（QSpinBox / QDoubleSpinBox / QCheckBox / QLineEdit / QComboBox），
        并绑定值变化时的回调。
        """
        self._clear_props_form()
        if node is None:
            self._props_layout.addRow(QLabel(ZH.PROP_SELECT_NODE))
            return
        self._props_node_id = node.id
        fields = EDITABLE_FIELDS.get(node.type, [])
        if not fields:
            # 该类型无可编辑字段（如 Sigmoid/Add），仅显示代码预览
            self._props_layout.addRow(QLabel(ZH.PROP_NO_FIELDS))
            self._code_preview.setPlainText(code_preview_for_node(node))
            self._sync_code_panel(node)
            return

        self._props_loading = True  # 防止表单加载过程中触发回调
        params = node.params
        for spec in fields:
            if len(spec) == 3:
                key, kind, choices = spec  # type: ignore[misc]
            else:
                key, kind = spec  # type: ignore[misc]
                choices = None
            label = param_label_zh(key)
            # 根据字段类型创建不同的 Qt 控件
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
            elif kind == "choice":
                w = QComboBox()
                chs = tuple(choices) if choices else ()
                for ch in chs:
                    w.addItem(ch)
                cur = str(params.get(key, chs[0] if chs else ""))
                idx = w.findText(cur)
                w.setCurrentIndex(max(0, idx))
                w.currentTextChanged.connect(lambda t, k=key: self._on_prop_choice_changed(k, t))
                self._props_layout.addRow(label, w)
        self._props_loading = False
        # 更新代码预览
        self._code_preview.setPlainText(code_preview_for_node(node))
        self._sync_code_panel(node)

    def _apply_code_from_preview(self) -> None:
        if self._code_preview.isReadOnly():
            return
        c = self._current_canvas()
        if c is None or self._props_node_id is None:
            return
        doc = c.graph_document()
        n = doc.get_node(self._props_node_id)
        if n is None:
            return
        patch, err = apply_code_preview_text(n.type, self._code_preview.toPlainText())
        if err:
            QMessageBox.warning(self, ZH.MSG_CODE_APPLY_FAIL_TITLE, err)
            return
        self._apply_param(patch)
        n2 = doc.get_node(self._props_node_id)
        if n2 is not None:
            self._fill_props_form(n2)

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

    def _on_prop_choice_changed(self, key: str, text: str) -> None:
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
        n2 = doc.get_node(self._props_node_id)
        if n2 is not None:
            self._code_preview.setPlainText(code_preview_for_node(n2))

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

    def _input_chw_from_doc(self, doc: GraphDocument) -> tuple[int, int, int]:
        from dl_vis.model.node_types import NodeType

        for n in doc.iter_nodes():
            if n.type == NodeType.INPUT.value:
                p = n.params
                return (
                    int(p.get("channels", 3)),
                    int(p.get("height", 224)),
                    int(p.get("width", 224)),
                )
        return 3, 224, 224

    def _run_dummy_forward(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        doc = c.graph_document()
        chk = check_graph_runnable(doc)
        if not chk.ok:
            QMessageBox.warning(self, ZH.RUN_FORWARD_FAIL, chk.error or chk.shape.message)
            return
        try:
            out = dummy_forward(doc)
        except ImportError as e:
            QMessageBox.warning(self, ZH.RUN_TORCH_HINT, str(e))
            return
        except ValueError as e:
            QMessageBox.warning(self, ZH.RUN_FORWARD_FAIL, str(e))
            return
        lines = [chk.shape.message, "", ZH.SHAPE_SUPPORT_NOTE]
        if chk.shape.warnings:
            lines.extend(["", "警告："])
            lines.extend(chk.shape.warnings)
        lines.extend(
            [
                "",
                "输出张量：",
                f"  shape = {out['output_shape']}",
                f"  mean = {out['mean']:.6g}",
                f"  std = {out['std']:.6g}",
                f"  min = {out['min']:.6g}",
                f"  max = {out['max']:.6g}",
            ]
        )
        QMessageBox.information(self, ZH.DLG_DUMMY_FORWARD, "\n".join(lines))

    def _open_train_dialog(self) -> None:
        """
        打开训练对话框。

        用户在对话框中可选择：
        1. 数据模式（画布数据集 / 合成 / 双 .npy / CSV）
        2. 训练轮数和学习率
        3. 开始训练（在后 QTbread 中运行，实时显示损失）
        """
        c = self._current_canvas()
        if c is None:
            return
        doc = c.graph_document()
        # 训练前的校验检查
        pre = validate_output_for_train(doc)
        if pre:
            QMessageBox.warning(self, ZH.RUN_TRAIN_FAIL, pre)
            return
        chk = check_graph_runnable(doc)
        if not chk.ok:
            QMessageBox.warning(self, ZH.RUN_TRAIN_FAIL, chk.error or chk.shape.message)
            return
        if self._train_worker is not None and self._train_worker.isRunning():
            QMessageBox.information(self, ZH.MSG_HINT_TITLE, ZH.TRAIN_BUSY)
            return

        from dl_vis.logic.graph_dataset import parse_graph_linked_training

        # 构建训练对话框 UI
        dlg = QDialog(self)
        dlg.setWindowTitle(ZH.DLG_TRAIN)
        root = QVBoxLayout(dlg)
        form = QFormLayout()

        # 数据模式选择
        mode_combo = QComboBox()
        if parse_graph_linked_training(doc) is not None:
            mode_combo.addItem(ZH.TRAIN_MODE_GRAPH, "graph")
        mode_combo.addItem(ZH.TRAIN_MODE_SYNTHETIC, "synthetic")
        mode_combo.addItem(ZH.TRAIN_MODE_NPY, "npy")
        mode_combo.addItem(ZH.TRAIN_MODE_CSV, "csv")
        form.addRow(ZH.TRAIN_DATA_MODE, mode_combo)

        # 训练超参
        sp_ep = QSpinBox()
        sp_ep.setRange(1, 100_000)
        sp_ep.setValue(5)
        form.addRow(ZH.TRAIN_EPOCHS, sp_ep)

        sp_lr = QDoubleSpinBox()
        sp_lr.setDecimals(8)
        sp_lr.setRange(1e-12, 1.0)
        sp_lr.setValue(1e-3)
        form.addRow(ZH.TRAIN_LR, sp_lr)

        # 双 .npy 文件选择区域
        ed_x = QLineEdit()
        ed_y = QLineEdit()
        bx = QPushButton("…")
        by = QPushButton("…")

        def browse_x() -> None:
            p, _ = QFileDialog.getOpenFileName(dlg, ZH.TRAIN_PATH_X, "", "NumPy (*.npy);;所有文件 (*)")
            if p:
                ed_x.setText(p)

        def browse_y() -> None:
            p, _ = QFileDialog.getOpenFileName(dlg, ZH.TRAIN_PATH_Y, "", "NumPy (*.npy);;所有文件 (*)")
            if p:
                ed_y.setText(p)

        bx.clicked.connect(browse_x)
        by.clicked.connect(browse_y)
        row_npy = QHBoxLayout()
        row_npy.addWidget(QLabel("X"))
        row_npy.addWidget(ed_x, 1)
        row_npy.addWidget(bx)
        row_npy.addWidget(QLabel("y"))
        row_npy.addWidget(ed_y, 1)
        row_npy.addWidget(by)
        npy_widget = QWidget()
        npy_widget.setLayout(row_npy)

        # CSV 文件选择区域
        ed_csv = QLineEdit()
        bc = QPushButton("…")
        chk_hdr = QCheckBox(ZH.TRAIN_CSV_SKIP_HEADER)

        def browse_csv() -> None:
            p, _ = QFileDialog.getOpenFileName(dlg, ZH.TRAIN_PATH_CSV, "", "CSV (*.csv);;所有文件 (*)")
            if p:
                ed_csv.setText(p)

        bc.clicked.connect(browse_csv)
        row_csv = QHBoxLayout()
        row_csv.addWidget(ed_csv, 1)
        row_csv.addWidget(bc)
        csv_widget = QWidget()
        csv_l = QVBoxLayout(csv_widget)
        csv_l.addLayout(row_csv)
        csv_l.addWidget(chk_hdr)

        root.addLayout(form)
        root.addWidget(npy_widget)
        root.addWidget(csv_widget)

        # 训练说明
        note = QLabel(ZH.TRAIN_SOFTMAX_NOTE)
        note.setWordWrap(True)
        root.addWidget(note)

        # 训练日志显示区域
        log = QTextEdit()
        log.setReadOnly(True)
        log.setMinimumHeight(180)
        root.addWidget(QLabel(ZH.TRAIN_LOG_HEADER))
        root.addWidget(log, 1)

        # 开始训练和关闭按钮
        btn_start = QPushButton(ZH.TRAIN_START)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        row_btn = QHBoxLayout()
        row_btn.addWidget(btn_start)
        row_btn.addStretch(1)
        row_btn.addWidget(bb)
        root.addLayout(row_btn)

        # 同步控件可见性：当模式切换时启用/禁用对应的输入区域
        def sync_mode() -> None:
            m = mode_combo.currentData()
            npy_widget.setEnabled(m == "npy")
            csv_widget.setEnabled(m == "csv")

        mode_combo.currentIndexChanged.connect(lambda _i: sync_mode())
        sync_mode()

        def start_train() -> None:
            """训练按钮点击回调：校验输入 -> 创建 TrainingWorker -> 启动后台线程。"""
            try:
                import torch  # noqa: F401
            except ImportError:
                QMessageBox.warning(dlg, ZH.RUN_TORCH_HINT, MSG_TORCH_MISSING)
                return
            mode = str(mode_combo.currentData())
            # 根据模式校验必要文件路径
            if mode == "npy":
                if not ed_x.text().strip() or not ed_y.text().strip():
                    QMessageBox.warning(dlg, ZH.RUN_TRAIN_FAIL, "请选择 X 与 y 的 .npy 文件路径。")
                    return
            if mode == "csv":
                if not ed_csv.text().strip():
                    QMessageBox.warning(dlg, ZH.RUN_TRAIN_FAIL, "请选择 CSV 文件路径。")
                    return
            ch, hh, ww = self._input_chw_from_doc(doc)
            # 构建训练任务配置
            cfg = TrainingJobConfig(
                epochs=int(sp_ep.value()),
                lr=float(sp_lr.value()),
                mode=mode,  # type: ignore[arg-type]
                x_path=ed_x.text().strip(),
                y_path=ed_y.text().strip(),
                csv_path=ed_csv.text().strip(),
                csv_skip_header=chk_hdr.isChecked(),
                channels=ch,
                height=hh,
                width=ww,
            )
            log.clear()
            # 创建后台训练线程
            worker = TrainingWorker(doc, cfg, self)
            self._train_worker = worker

            def on_ep(ep: int, lv: float) -> None:
                """每个 epoch 结束的回调：在 UI 日志区追加当前损失。"""
                log.append(f"epoch {ep}: loss = {lv:.6g}")

            def on_ok(losses: object) -> None:
                """训练成功完成。"""
                self._train_worker = None
                worker.deleteLater()
                btn_start.setEnabled(True)
                if isinstance(losses, list):
                    lines = [f"epoch {i + 1}: {float(v):.6g}" for i, v in enumerate(losses)]
                    QMessageBox.information(dlg, ZH.DLG_TRAIN_RESULT, "\n".join(lines[:200]) + ("\n…" if len(lines) > 200 else ""))

            def on_fail(msg: str) -> None:
                """训练失败。"""
                self._train_worker = None
                worker.deleteLater()
                btn_start.setEnabled(True)
                QMessageBox.warning(dlg, ZH.RUN_TRAIN_FAIL, msg)

            # 连接信号与槽
            worker.epoch_loss.connect(on_ep)
            worker.finished_ok.connect(on_ok)
            worker.failed.connect(on_fail)
            btn_start.setEnabled(False)
            worker.start()

        btn_start.clicked.connect(start_train)
        dlg.resize(560, 480)
        dlg.exec()

    def _run_shape_inference(self) -> None:
        c = self._current_canvas()
        if c is None:
            return
        res = infer_shapes_dag_nchw(c.graph_document())
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
