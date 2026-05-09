"""QApplication 入口。"""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import QLocale, Qt
from PyQt6.QtWidgets import QApplication

from dl_vis.logging_config import (
    enable_faulthandler,
    install_excepthook,
    install_qt_message_handler,
    setup_logging,
)
from dl_vis.ui.main_window import MainWindow


def main() -> None:
    setup_logging()
    enable_faulthandler()
    install_excepthook()
    log = logging.getLogger("dl_vis")
    log.info(
        "启动 dl_vis；调试请设 DL_VIS_DEBUG=1（在系统临时目录写入 dl_vis.log）或 DL_VIS_LOG_LEVEL=DEBUG"
    )

    QLocale.setDefault(QLocale(QLocale.Language.Chinese, QLocale.Country.China))

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    install_qt_message_handler()
    app.setApplicationName("dl_vis")
    app.setStyleSheet(
        "QToolTip { max-width: 520px; padding: 10px; white-space: pre-wrap; }"
    )

    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
