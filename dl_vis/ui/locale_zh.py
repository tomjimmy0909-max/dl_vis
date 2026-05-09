"""简体中文界面文案（集中维护）。"""

from __future__ import annotations

APP_TITLE_BASE = "dl_vis — 深度学习可视化建模"

DOCK_PALETTE = "节点类型"
DOCK_PARAMS = "参数"

MENU_FILE = "文件"
MENU_TOOL = "工具"
MENU_EXPORT = "导出"

ACTION_NEW_TAB = "新建画布页"
ACTION_OPEN = "打开…"
ACTION_SAVE = "保存…"
ACTION_COPY_NODES = "复制选中节点"
ACTION_SHAPE_INFER = "推导形状（链式占位）…"
ACTION_EXPORT_TORCH = "导出为 PyTorch…"

TAB_EDITOR = "图形编辑"
TAB_VIZ = "可视化"
TAB_CANVAS_N = "画布 {}"

MSG_KEEP_ONE_TAB = "至少保留一个画布页。"
MSG_HINT_TITLE = "提示"

PROP_SELECT_NODE = "（选中一个节点以编辑参数）"
PROP_NO_FIELDS = "（该类型无可编辑字段）"

DLG_OPEN_GRAPH = "打开图"
DLG_SAVE_GRAPH = "保存图"
FILE_FILTER_JSON = "JSON (*.json)"
FILE_FILTER_ALL = "所有文件 (*)"

MSG_OPEN_FAIL = "打开失败"
MSG_PARSE_FAIL = "解析失败"
MSG_SAVE_FAIL = "保存失败"

DLG_EXPORT_TORCH = "导出 PyTorch"
DLG_SHAPE_INFER = "形状推导（占位）"

SHAPE_ROW_PREFIX = "  {}：NCHW = {}"

CANVAS_WIRE_TO_IN_PORT = "请将连线拖到目标节点的输入端口（左侧蓝色圆点）。"
CANVAS_WIRE_ADDED = "已添加连线。"

MENU_EDIT = "编辑"
ACTION_UNDO = "撤销"
ACTION_REDO = "重做"
ACTION_ALIGN_LEFT = "左对齐"
ACTION_ALIGN_TOP = "顶对齐"

ALIGN_NEED_TWO = "请至少选中两个节点后再对齐。"

UNDO_ADD_NODE = "添加节点"
UNDO_DELETE = "删除"
UNDO_ADD_EDGE = "添加连线"
UNDO_DUPLICATE = "复制节点"
UNDO_MOVE_NODE = "移动节点"
UNDO_ALIGN = "对齐节点"
UNDO_EDIT_PARAM = "修改参数"

ACTION_EXPORT_COPY = "复制 Sequential 源码"
ACTION_EXPORT_SAVE = "导出 Sequential 为 .py…"

EXPORT_FAIL_TITLE = "导出失败"
SHAPE_SUPPORT_NOTE = "当前形状推导支持范围请见对话框顶部说明。"
