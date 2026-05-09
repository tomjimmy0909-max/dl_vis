"""PyTorch 导出：线性 nn.Sequential 源码生成（分支/残差等拓扑不支持）。"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from dl_vis.model.node_types import NodeType

if TYPE_CHECKING:
    from dl_vis.model.graph_document import GraphDocument


def export_stub_message() -> str:
    return (
        "完整导出（含分支、残差、Attention）计划在后续阶段实现。\n"
        "当前可使用菜单「导出 → 复制 Sequential 源码 / 导出为 .py」生成单路径线性链 nn.Sequential。"
    )


def export_to_torch_module(document: GraphDocument) -> Any:
    """尝试构建内存中的 nn.Sequential（仅线性链）。"""
    import torch.nn as nn

    src = export_sequential_source(document, module_var="seq")
    g: dict[str, Any] = {}
    exec(src, {"nn": nn, "torch": __import__("torch")}, g)
    return g["seq"]


def linear_chain_order(doc: GraphDocument) -> list[str]:
    """验证图为一条覆盖全部节点的有向路径；返回从唯一起点到终点的节点 id 序列。"""
    nodes = set(doc.nodes.keys())
    if len(nodes) == 0:
        raise ValueError("图为空，无法导出。")
    indeg: dict[str, int] = defaultdict(int)
    succ: dict[str, str | None] = {nid: None for nid in nodes}
    pred: dict[str, str | None] = {nid: None for nid in nodes}
    for e in doc.edges.values():
        if e.src_id not in nodes or e.dst_id not in nodes:
            raise ValueError("边引用了不存在的节点。")
        indeg[e.dst_id] += 1
        if succ[e.src_id] is not None:
            raise ValueError("存在分支（一个节点有多条出边），Sequential 无法表达。")
        succ[e.src_id] = e.dst_id
        if pred[e.dst_id] is not None:
            raise ValueError("存在合并（一个节点有多条入边），Sequential 无法表达。")
        pred[e.dst_id] = e.src_id

    roots = [nid for nid in nodes if indeg[nid] == 0]
    if len(roots) != 1:
        raise ValueError("线性链要求恰好一个入度为 0 的起点（通常为 Input）。")
    cur = roots[0]
    order: list[str] = []
    seen: set[str] = set()
    while cur is not None:
        if cur in seen:
            raise ValueError("检测到环。")
        seen.add(cur)
        order.append(cur)
        cur = succ[cur]
    if len(order) != len(nodes):
        raise ValueError("图不是覆盖全部节点的单一路径（可能存在孤立点或多分量）。")
    if len(doc.edges) != max(0, len(order) - 1):
        raise ValueError("边数与线性路径不一致。")
    return order


def _repr_bool(b: bool) -> str:
    return "True" if b else "False"


def export_sequential_source(doc: GraphDocument, *, module_var: str = "net") -> str:
    """生成可执行的 Python 源码字符串：``import torch.nn`` + ``module_var = nn.Sequential(...)``。"""
    order = linear_chain_order(doc)
    layers: list[str] = []
    spatial = True  # 是否在卷积特征图上（需 Flatten 再 Linear）

    for nid in order:
        n = doc.get_node(nid)
        if n is None:
            raise ValueError("节点缺失。")
        t = n.type
        p = copy.deepcopy(n.params)

        if t == NodeType.INPUT.value:
            continue
        if t == NodeType.OUTPUT.value:
            continue
        if t in (NodeType.RESIDUAL.value, NodeType.PRUNE.value, NodeType.ATTENTION.value):
            raise ValueError(f"节点类型「{t}」当前不支持导出为 Sequential。")

        if t == NodeType.CONV3X3.value:
            ic = int(p["in_channels"])
            oc = int(p["out_channels"])
            s = int(p.get("stride", 1))
            pad = int(p.get("padding", 0))
            bias = bool(p.get("bias", True))
            layers.append(f"nn.Conv2d({ic}, {oc}, kernel_size=3, stride={s}, padding={pad}, bias={_repr_bool(bias)})")
            spatial = True
        elif t == NodeType.CONV1X1.value:
            ic = int(p["in_channels"])
            oc = int(p["out_channels"])
            s = int(p.get("stride", 1))
            pad = int(p.get("padding", 0))
            bias = bool(p.get("bias", True))
            layers.append(f"nn.Conv2d({ic}, {oc}, kernel_size=1, stride={s}, padding={pad}, bias={_repr_bool(bias)})")
            spatial = True
        elif t == NodeType.MAX_POOL.value:
            ks = int(p.get("kernel_size", 2))
            st = int(p.get("stride", ks))
            pad = int(p.get("padding", 0))
            layers.append(f"nn.MaxPool2d(kernel_size={ks}, stride={st}, padding={pad})")
            spatial = True
        elif t == NodeType.AVG_POOL.value:
            ks = int(p.get("kernel_size", 2))
            st = int(p.get("stride", ks))
            pad = int(p.get("padding", 0))
            layers.append(f"nn.AvgPool2d(kernel_size={ks}, stride={st}, padding={pad})")
            spatial = True
        elif t == NodeType.FC.value:
            inf = int(p["in_features"])
            outf = int(p["out_features"])
            bias = bool(p.get("bias", True))
            if spatial:
                layers.append("nn.Flatten()")
                spatial = False
            layers.append(f"nn.Linear({inf}, {outf}, bias={_repr_bool(bias)})")
        elif t == NodeType.RELU.value:
            inplace = bool(p.get("inplace", False))
            layers.append(f"nn.ReLU(inplace={_repr_bool(inplace)})")
        elif t == NodeType.SIGMOID.value:
            layers.append("nn.Sigmoid()")
        elif t == NodeType.SOFTMAX.value:
            dim = int(p.get("dim", -1))
            layers.append(f"nn.Softmax(dim={dim})")
        elif t == NodeType.BN.value:
            nf = int(p["num_features"])
            eps = float(p.get("eps", 1e-5))
            momentum = float(p.get("momentum", 0.1))
            affine = bool(p.get("affine", True))
            layers.append(
                f"nn.BatchNorm2d({nf}, eps={eps}, momentum={momentum}, affine={_repr_bool(affine)})"
            )
            spatial = True
        else:
            raise ValueError(f"未支持的节点类型: {t}")

    if not layers:
        raise ValueError("没有可导出的计算层（是否仅有 Input/Output？）。")

    joined = ",\n    ".join(layers)
    return (
        "import torch\n"
        "import torch.nn as nn\n\n"
        f"{module_var} = nn.Sequential(\n    {joined},\n)\n"
    )
