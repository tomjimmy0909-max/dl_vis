"""从图中解析「Dataset → Input」训练数据绑定，供导出与内存训练共用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dl_vis.model.graph_document import GraphDocument, GraphNode
from dl_vis.model.node_types import NodeType

GraphTrainMode = Literal["image_folder", "csv", "npy_pair"]


@dataclass
class GraphLinkedTraining:
    """单条从画布解析出的训练数据规格。"""

    mode: GraphTrainMode
    primary: str
    secondary: str = ""
    csv_skip_header: bool = False


def _dataset_path_and_kind(n: GraphNode) -> tuple[str, str]:
    p = str(n.params.get("path", "")).strip()
    kind = str(n.params.get("path_kind", "file")).strip().lower()
    path_obj = Path(p) if p else None
    if path_obj is not None and p and path_obj.exists():
        kind = "folder" if path_obj.is_dir() else "file"
    return p, kind


def datasets_feeding_input(doc: GraphDocument) -> list[GraphNode]:
    """所有存在边 Dataset → Input 的数据集节点（去重、顺序稳定）。"""
    inputs = [n for n in doc.iter_nodes() if n.type == NodeType.INPUT.value]
    if len(inputs) != 1:
        return []
    inp_id = inputs[0].id
    seen: set[str] = set()
    out: list[GraphNode] = []
    for e in sorted(doc.edges.values(), key=lambda x: (x.src_id, x.dst_id)):
        if e.dst_id != inp_id:
            continue
        n = doc.get_node(e.src_id)
        if n is None or n.type != NodeType.DATASET.value:
            continue
        if n.id in seen:
            continue
        seen.add(n.id)
        out.append(n)
    return out


def parse_graph_linked_training(doc: GraphDocument) -> GraphLinkedTraining | None:
    """
    解析 Dataset→Input 以决定训练数据形态。
    - 单路径且为目录（或 path_kind=folder）：ImageFolder。
    - 单路径且 ``.csv``：按现有 CSV NCHW+标签行协议加载。
    - 两条边且均为 ``.npy``：按 ``(X,Y)`` 加载（路径按字典序对应，首为 X、次为 Y）。
    """
    nodes = datasets_feeding_input(doc)
    if not nodes:
        return None
    specs: list[tuple[str, str, GraphNode]] = []
    for n in nodes:
        p, k = _dataset_path_and_kind(n)
        if not p:
            continue
        specs.append((p, k, n))
    if not specs:
        return None

    if len(specs) >= 2:
        npy_specs = [t for t in specs if t[0].lower().endswith(".npy")]
        if len(npy_specs) >= 2:
            paths = sorted(t[0] for t in npy_specs)
            return GraphLinkedTraining(mode="npy_pair", primary=paths[0], secondary=paths[1])

    p, kind, _n = specs[0]
    path_obj = Path(p)
    if kind == "folder" or path_obj.is_dir():
        return GraphLinkedTraining(mode="image_folder", primary=str(path_obj.resolve()))
    if p.lower().endswith(".csv"):
        _pk, _kk, node0 = specs[0]
        skip = bool(node0.params.get("csv_skip_header", False))
        return GraphLinkedTraining(
            mode="csv", primary=str(path_obj.resolve()), csv_skip_header=skip
        )
    if p.lower().endswith(".npy"):
        return None
    return None


def describe_graph_training_hint(doc: GraphDocument) -> str:
    """用于 UI 提示：当前图上训练数据为何不可用。"""
    if parse_graph_linked_training(doc) is not None:
        return ""
    nodes = datasets_feeding_input(doc)
    if not nodes:
        return "请添加「训练数据集」节点，并连接：**数据集出口 → 输入层入口**。"
    paths = [_dataset_path_and_kind(n)[0] for n in nodes]
    if not any(paths):
        return "数据集节点请填写有效 path。"
    if len(nodes) == 1 and paths[0].lower().endswith(".npy"):
        return "单个 .npy 无法区分 X/y；请再添加一个数据集节点并连到 Input，或使用文件夹 / CSV。"
    return "当前路径组合无法自动识别为 文件夹(ImageFolder)、CSV 或 双 .npy；请检查路径。"
