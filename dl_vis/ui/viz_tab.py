"""matplotlib 可视化：选中节点的数值型参数条形图（占位扩展）。"""

from __future__ import annotations

import logging

import matplotlib

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from dl_vis.ui.node_item import NodeItem

_LOG = logging.getLogger(__name__)

_MPL_FONT_CONFIGURED = False


def _configure_matplotlib_cjk_font() -> None:
    """Windows/Linux/macOS 常见中文字体优先，避免 DejaVu Sans 缺字警告。"""
    global _MPL_FONT_CONFIGURED
    if _MPL_FONT_CONFIGURED:
        return
    _MPL_FONT_CONFIGURED = True
    sans = matplotlib.rcParams["font.sans-serif"]
    if not isinstance(sans, list):
        sans = list(sans)
    preferred = [
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "KaiTi",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    matplotlib.rcParams["font.sans-serif"] = preferred + [f for f in sans if f not in preferred]
    matplotlib.rcParams["axes.unicode_minus"] = False


class VizTab(QWidget):
    """依赖 MainWindow._current_canvas() 与节点选中状态。"""

    def __init__(self, main_window: QWidget) -> None:
        super().__init__()
        self._mw = main_window
        lay = QVBoxLayout(self)
        self._hint = QLabel("")
        lay.addWidget(self._hint)
        self._figure = Figure(figsize=(6, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)
        lay.addWidget(self._canvas)
        self.refresh_from_main()

    def _draw_idle_safe(self) -> None:
        try:
            self._canvas.draw_idle()
        except Exception:
            _LOG.exception("FigureCanvas.draw_idle 失败，回退 draw")
            self._canvas.draw()

    def _show_error_axes(self, msg: str) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)
        self._draw_idle_safe()

    def refresh_from_main(self) -> None:
        """刷新图表；内部捕获异常，避免拖垮整个应用。"""
        try:
            self._refresh_from_main_impl()
        except Exception as e:
            _LOG.exception("可视化刷新失败")
            self._hint.setText("刷新出错，请查看 stderr 日志。")
            self._show_error_axes(f"刷新失败：{e}")

    def _refresh_from_main_impl(self) -> None:
        _configure_matplotlib_cjk_font()
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        get_canvas = getattr(self._mw, "_current_canvas", None)
        if get_canvas is None:
            self._hint.setText("")
            ax.text(0.5, 0.5, "内部错误：无法访问画布。", ha="center", va="center")
            self._draw_idle_safe()
            return
        c = get_canvas()
        if c is None:
            self._hint.setText("")
            ax.text(0.5, 0.5, "无活动画布。", ha="center", va="center")
            self._draw_idle_safe()
            return

        sel = [it for it in c.scene().selectedItems() if isinstance(it, NodeItem)]
        if not sel:
            self._hint.setText("提示：在「图形编辑」画布中选中一个节点以查看其数值参数分布。")
            ax.text(0.5, 0.5, "未选中节点", ha="center", va="center")
            self._draw_idle_safe()
            return

        node = c.graph_document().get_node(sel[0].node_id)
        if node is None:
            ax.text(0.5, 0.5, "节点不存在", ha="center", va="center")
            self._draw_idle_safe()
            return

        self._hint.setText(f"当前节点：{node.type}（显示数值型 params）")
        pairs: list[tuple[str, float]] = []
        for k, v in node.params.items():
            if isinstance(v, bool):
                pairs.append((k, 1.0 if v else 0.0))
            elif isinstance(v, (int, float)):
                pairs.append((k, float(v)))

        if not pairs:
            ax.text(0.5, 0.5, "该节点无可绘制的数值参数。", ha="center", va="center")
            self._draw_idle_safe()
            return

        pairs.sort(key=lambda x: x[0])
        labels = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("值（布尔映射为 0/1）")
        ax.set_title(f"{node.type} 参数")
        try:
            self._figure.tight_layout()
        except Exception:
            _LOG.debug("tight_layout 跳过", exc_info=True)
        self._draw_idle_safe()
