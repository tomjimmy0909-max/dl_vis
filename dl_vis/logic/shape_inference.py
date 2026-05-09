"""形状推导占位：简单线性链 NCHW；分支/合并返回未实现提示。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dl_vis.model.graph_document import GraphDocument

from dl_vis.model.node_types import NodeType


@dataclass
class ShapeResult:
    ok: bool
    message: str
    shapes_by_node: dict[str, tuple[int, int, int, int]] | None = None  # N,C,H,W
    warnings: list[str] = field(default_factory=list)


def _topological_order(doc: GraphDocument) -> list[str] | None:
    """Kahn 拓扑排序；若有环返回 None（文档层应已无环）。"""
    nodes = set(doc.nodes.keys())
    if not nodes:
        return []
    in_deg = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for e in doc.edges.values():
        if e.src_id in adj and e.dst_id in in_deg:
            adj[e.src_id].append(e.dst_id)
            in_deg[e.dst_id] += 1
    queue = [n for n in nodes if in_deg[n] == 0]
    order: list[str] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    if len(order) != len(nodes):
        return None
    return order


def infer_shapes_linear_nchw(doc: GraphDocument) -> ShapeResult:
    """
    从 Input 节点出发，沿唯一拓扑后继传播 NCHW。
    若有多个 Input、分支合并或非链式拓扑，返回 ok=False 与说明。
    """
    inputs = [n for n in doc.iter_nodes() if n.type == NodeType.INPUT.value]
    if len(inputs) != 1:
        return ShapeResult(
            ok=False,
            message="当前占位推导仅支持图中恰好一个 Input 节点。",
            shapes_by_node=None,
        )
    root_id = inputs[0].id
    order = _topological_order(doc)
    if order is None:
        return ShapeResult(ok=False, message="图中存在环，无法推导。", shapes_by_node=None)
    _ = order  # 占位：完整推导可按拓扑序遍历；链式路径单独 walk

    # 每个节点的入边数量（用于检测链式）
    indeg: dict[str, int] = {nid: 0 for nid in doc.nodes}
    outdeg: dict[str, int] = {nid: 0 for nid in doc.nodes}
    succ: dict[str, str | None] = {nid: None for nid in doc.nodes}
    pred: dict[str, str | None] = {nid: None for nid in doc.nodes}
    for e in doc.edges.values():
        indeg[e.dst_id] += 1
        outdeg[e.src_id] += 1
        if succ[e.src_id] is not None:
            return ShapeResult(
                ok=False,
                message=(
                    "存在分支（一个节点有多条出边）。\n"
                    "【一档】线性链推导不适用；【二档】多分支逐路径推导与汇合尚未在本版本实现。"
                ),
                shapes_by_node=None,
            )
        succ[e.src_id] = e.dst_id
        if pred[e.dst_id] is not None:
            return ShapeResult(
                ok=False,
                message=(
                    "存在合并（一个节点有多条入边）。\n"
                    "【一档】线性链推导不适用；【二档】汇合点语义推导尚未在本版本实现。"
                ),
                shapes_by_node=None,
            )
        pred[e.dst_id] = e.src_id

    if indeg.get(root_id, 0) != 0:
        return ShapeResult(ok=False, message="Input 节点不应有入边。", shapes_by_node=None)

    shapes: dict[str, tuple[int, int, int, int]] = {}
    warnings: list[str] = []
    cur_id: str | None = root_id
    inp = doc.get_node(root_id)
    if inp is None:
        return ShapeResult(ok=False, message="找不到 Input。", shapes_by_node=None)
    p = inp.params
    n, c, h, w = int(p.get("batch", 1)), int(p.get("channels", 3)), int(p.get("height", 224)), int(p.get("width", 224))
    shapes[root_id] = (n, c, h, w)

    while cur_id is not None:
        nxt = succ.get(cur_id)
        if nxt is None:
            break
        parent = doc.get_node(cur_id)
        child = doc.get_node(nxt)
        if parent is None or child is None:
            return ShapeResult(ok=False, message="边引用缺失节点。", shapes_by_node=None)
        nc, cc, hc, wc = shapes[cur_id]

        t = child.type
        if t in (NodeType.RELU.value, NodeType.SIGMOID.value, NodeType.SOFTMAX.value, NodeType.BN.value):
            shapes[nxt] = (nc, cc, hc, wc)
        elif t == NodeType.CONV3X3.value or t == NodeType.CONV1X1.value:
            cp = child.params
            oc = int(cp.get("out_channels", cc))
            stride = int(cp.get("stride", 1))
            pad = int(cp.get("padding", 0))
            kh = 3 if t == NodeType.CONV3X3.value else 1
            kw = kh
            nh = (hc + 2 * pad - kh) // stride + 1
            nw = (wc + 2 * pad - kw) // stride + 1
            shapes[nxt] = (nc, oc, nh, nw)
        elif t == NodeType.MAX_POOL.value or t == NodeType.AVG_POOL.value:
            cp = child.params
            ks = int(cp.get("kernel_size", 2))
            stride = int(cp.get("stride", ks))
            pad = int(cp.get("padding", 0))
            nh = (hc + 2 * pad - ks) // stride + 1
            nw = (wc + 2 * pad - ks) // stride + 1
            shapes[nxt] = (nc, cc, nh, nw)
        elif t == NodeType.FC.value:
            flat = cc * hc * wc
            inf = int(child.params.get("in_features", flat))
            outf = int(child.params.get("out_features", 10))
            if inf != flat:
                warnings.append(
                    f"FC 节点（id 前缀 {nxt[:8]}…）：in_features={inf} 与上游展平维 C×H×W={flat} 不一致（占位校验）。"
                )
            shapes[nxt] = (nc, outf, 1, 1)
        elif t in (
            NodeType.INPUT.value,
            NodeType.OUTPUT.value,
            NodeType.RESIDUAL.value,
            NodeType.PRUNE.value,
            NodeType.ATTENTION.value,
        ):
            shapes[nxt] = (nc, cc, hc, wc)
        else:
            return ShapeResult(
                ok=False,
                message=f"未知或未建模算子类型: {t}（可在第二阶段扩展传播规则）。",
                shapes_by_node=None,
            )
        cur_id = nxt

    msg = (
        "【一档】线性链 NCHW 推导完成（单 Input、无分支/汇合）。\n"
        "【二档】单分叉多路径 / 同形状 element-wise 汇合：尚未实现。\n"
        "【三档】任意 DAG 完整推导：规划中。"
    )
    return ShapeResult(ok=True, message=msg, shapes_by_node=shapes, warnings=warnings)
