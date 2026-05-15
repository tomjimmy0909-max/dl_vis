"""DAG 拓扑排序与环检测（Kahn），供形状推导、代码生成等复用。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dl_vis.model.graph_document import GraphDocument


def has_cycle(doc: GraphDocument) -> bool:
    """若图中存在有向环返回 ``True``。"""
    if not doc.nodes:
        return False
    return topological_sort(doc) is None  # 拓扑排序失败说明存在环


def topological_sort(doc: GraphDocument) -> list[str] | None:
    """
    Kahn 拓扑排序；返回覆盖全部节点 id 的序列；若存在环返回 ``None``。

    算法原理：
    1. 统计每个节点的入度（指向它的边数）
    2. 将入度为 0 的节点加入队列
    3. 依次出队并「删除」其出边（降低后继的入度）
    4. 若最终序列未覆盖全部节点，说明存在环
    """
    nodes = set(doc.nodes.keys())
    if not nodes:
        return []
    # 初始化入度表和邻接表
    in_deg = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    # 根据边信息构建邻接关系和入度
    for e in doc.edges.values():
        if e.src_id in adj and e.dst_id in in_deg:
            adj[e.src_id].append(e.dst_id)
            in_deg[e.dst_id] += 1
    # 收集所有入度为 0 的起点
    queue = [n for n in nodes if in_deg[n] == 0]
    order: list[str] = []
    while queue:
        u = queue.pop(0)        # 取出一个起点
        order.append(u)
        for v in adj[u]:
            in_deg[v] -= 1       # 模拟删除该边
            if in_deg[v] == 0:   # 入度归零则加入队列
                queue.append(v)
    # 若排序结果未覆盖所有节点，说明存在环
    if len(order) != len(nodes):
        return None
    return order
