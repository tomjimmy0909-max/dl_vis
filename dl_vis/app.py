"""QApplication 入口。"""

from __future__ import annotations

import sys

from PyQt6.QtCore import QLocale, Qt
from PyQt6.QtWidgets import QApplication

from dl_vis.ui.main_window import MainWindow


def main() -> None:
    QLocale.setDefault(QLocale(QLocale.Language.Chinese, QLocale.Country.China))

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("dl_vis")
    app.setStyleSheet(
        "QToolTip { max-width: 520px; padding: 10px; white-space: pre-wrap; }"
    )

    win = MainWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
