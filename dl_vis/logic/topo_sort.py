"""DAG 拓扑排序与环检测（Kahn），供形状推导、代码生成等复用。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dl_vis.model.graph_document import GraphDocument


def has_cycle(doc: GraphDocument) -> bool:
    """若图中存在有向环返回 ``True``。"""
    if not doc.nodes:
        return False
    return topological_sort(doc) is None


def topological_sort(doc: GraphDocument) -> list[str] | None:
    """
    Kahn 拓扑排序；返回覆盖全部节点 id 的序列；若存在环返回 ``None``。
    """
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
