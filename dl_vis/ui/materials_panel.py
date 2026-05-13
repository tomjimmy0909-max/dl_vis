"""左侧「训练素材」：从系统拖入或通过按钮添加文件/文件夹，再拖到画布生成 Dataset 节点。"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dl_vis.ui import locale_zh as ZH

MIME_DATASET = "application/x-dlvis-dataset"

ROLE_PATH = Qt.ItemDataRole.UserRole
ROLE_KIND = Qt.ItemDataRole.UserRole + 1


class MaterialsList(QListWidget):
    """列表项可拖到画布，载荷为 JSON path + kind。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def mimeData(self, items: list[QListWidgetItem]):  # type: ignore[override]
        md = super().mimeData(items)
        if not items:
            return md
        path = items[0].data(ROLE_PATH)
        kind = items[0].data(ROLE_KIND)
        if isinstance(path, str) and path and isinstance(kind, str):
            payload = json.dumps({"path": path, "kind": kind}, ensure_ascii=False)
            md.setData(MIME_DATASET, QByteArray(payload.encode("utf-8")))
        return md

    def supportedDragActions(self) -> Qt.DropAction:
        return Qt.DropAction.CopyAction


class MaterialsPanel(QWidget):
    """素材栏：添加文件/文件夹；支持从资源管理器拖入列表。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        tip = QLabel(ZH.MATERIALS_PANEL_HINT)
        tip.setWordWrap(True)
        v.addWidget(tip)

        row = QHBoxLayout()
        self._btn_file = QPushButton(ZH.MATERIALS_ADD_FILE)
        self._btn_dir = QPushButton(ZH.MATERIALS_ADD_FOLDER)
        self._btn_file.clicked.connect(self._pick_file)
        self._btn_dir.clicked.connect(self._pick_folder)
        row.addWidget(self._btn_file)
        row.addWidget(self._btn_dir)
        v.addLayout(row)

        self._list = MaterialsList()
        v.addWidget(self._list, 1)

        self.setAcceptDrops(True)

    def _pick_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, ZH.MATERIALS_ADD_FILE, "", ZH.MATERIALS_FILE_FILTER)
        if path_str:
            self._append_path(Path(path_str), "file")

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, ZH.MATERIALS_ADD_FOLDER, "")
        if d:
            self._append_path(Path(d), "folder")

    def _append_path(self, path: Path, kind: str) -> None:
        resolved = str(path.resolve())
        name = path.name if path.name else resolved
        it = QListWidgetItem(f"{ZH.MATERIALS_LIST_LABEL}{name}")
        it.setData(ROLE_PATH, resolved)
        it.setData(ROLE_KIND, kind)
        it.setToolTip(resolved)
        self._list.addItem(it)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for u in event.mimeData().urls():
                if u.isLocalFile() and u.toLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for u in event.mimeData().urls():
            if not u.isLocalFile():
                continue
            lf = u.toLocalFile()
            if not lf:
                continue
            p = Path(lf)
            self._append_path(p, "folder" if p.is_dir() else "file")
        event.acceptProposedAction()
