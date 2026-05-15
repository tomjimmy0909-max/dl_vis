"""QApplication 入口。"""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import QLocale, Qt
from PyQt6.QtWidgets import QApplication

from dl_vis.error_report import report_dir
from dl_vis.logging_config import (
    enable_faulthandler,
    install_excepthook,
    install_qt_message_handler,
    install_thread_excepthook,
    setup_logging,
)
from dl_vis.ui.main_window import MainWindow


def main() -> None:
    """
    dl_vis 应用程序入口。

    启动流程：
    1. 初始化日志系统
    2. 启用 faulthandler（段错误时输出 Python 栈）
    3. 安装未捕获异常钩子（写入结构化 JSON 报告）
    4. 配置 Qt 应用程序设置（DPI、本地化、样式）
    5. 创建并显示主窗口
    6. 启动 Qt 事件循环
    """
    # === 启动前初始化 ===
    setup_logging()
    enable_faulthandler()
    install_excepthook()
    install_thread_excepthook()
    log = logging.getLogger("dl_vis")
    log.info(
        "启动 dl_vis；调试请设 DL_VIS_DEBUG=1（在系统临时目录写入 dl_vis.log）或 DL_VIS_LOG_LEVEL=DEBUG"
    )
    log.info(
        "异常 JSON 报告目录：%s（可用环境变量 DL_VIS_ERROR_REPORT_DIR 覆盖；"
        "主日志行关键字 DL_VIS_ERROR_REPORT）",
        report_dir(),
    )

    # === Qt 应用程序配置 ===
    QLocale.setDefault(QLocale(QLocale.Language.Chinese, QLocale.Country.China))
    # 高 DPI 缩放策略：PassThrough 让系统管理缩放，避免 Qt 额外放大导致模糊
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    install_qt_message_handler()  # 将 Qt 原生警告重定向到 Python logging
    app.setApplicationName("dl_vis")
    # 全局样式：增大 Tooltip 宽度以容纳中文字符说明
    app.setStyleSheet(
        "QToolTip { max-width: 520px; padding: 10px; white-space: pre-wrap; }"
    )

    # === 创建并显示主窗口 ===
    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
