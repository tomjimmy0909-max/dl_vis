"""计算图文档：节点、边、DAG 校验与 JSON 序列化。"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from dl_vis.model.node_types import default_params_for_type, is_known_type


SCHEMA_VERSION = "1.0"


@dataclass
class GraphNode:
    id: str
    type: str
    x: float = 0.0
    y: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: str
    src_id: str
    dst_id: str
    src_port: str = "out"
    dst_port: str = "in"


class GraphDocument:
    """有向图文档；边为有向 (src -> dst)，端口预留。"""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}

    @property
    def nodes(self) -> dict[str, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> dict[str, GraphEdge]:
        return self._edges

    def iter_nodes(self) -> Iterator[GraphNode]:
        return iter(self._nodes.values())

    def iter_edges(self) -> Iterator[GraphEdge]:
        return iter(self._edges.values())

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def add_node(
        self,
        node_type: str,
        x: float = 0.0,
        y: float = 0.0,
        node_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> GraphNode:
        nid = node_id or str(uuid.uuid4())
        p = default_params_for_type(node_type) if is_known_type(node_type) else {}
        if params:
            p.update(params)
        n = GraphNode(id=nid, type=node_type, x=x, y=y, params=p)
        self._nodes[nid] = n
        return n

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        del self._nodes[node_id]
        to_del = [eid for eid, e in self._edges.items() if e.src_id == node_id or e.dst_id == node_id]
        for eid in to_del:
            del self._edges[eid]

    def update_node_position(self, node_id: str, x: float, y: float) -> None:
        n = self._nodes.get(node_id)
        if n:
            n.x, n.y = x, y

    def update_node_params(self, node_id: str, params: dict[str, Any]) -> None:
        n = self._nodes.get(node_id)
        if n:
            n.params.update(params)

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        src_port: str = "out",
        dst_port: str = "in",
        edge_id: str | None = None,
    ) -> tuple[GraphEdge | None, str | None]:
        """返回 (edge, error_message)。error 非空表示未添加。"""
        if src_id == dst_id:
            return None, "不允许自环"
        if src_id not in self._nodes or dst_id not in self._nodes:
            return None, "端点节点不存在"
        dup = any(e.src_id == src_id and e.dst_id == dst_id for e in self._edges.values())
        if dup:
            return None, "边已存在"
        eid = edge_id or str(uuid.uuid4())
        edge = GraphEdge(id=eid, src_id=src_id, dst_id=dst_id, src_port=src_port, dst_port=dst_port)
        self._edges[eid] = edge
        if self._has_cycle():
            del self._edges[eid]
            return None, "添加该边会产生环路"
        return edge, None

    def remove_edge(self, edge_id: str) -> None:
        self._edges.pop(edge_id, None)

    def _has_cycle(self) -> bool:
        """DFS 检测环。"""
        adj: dict[str, list[str]] = {nid: [] for nid in self._nodes}
        for e in self._edges.values():
            adj[e.src_id].append(e.dst_id)
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(u: str) -> bool:
            visited.add(u)
            stack.add(u)
            for v in adj.get(u, []):
                if v not in visited:
                    if dfs(v):
                        return True
                elif v in stack:
                    return True
            stack.remove(u)
            return False

        for nid in self._nodes:
            if nid not in visited:
                if dfs(nid):
                    return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": [asdict(n) for n in self._nodes.values()],
            "edges": [asdict(e) for e in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphDocument:
        doc = cls()
        ver = data.get("schema_version", SCHEMA_VERSION)
        if ver != SCHEMA_VERSION:
            # 第一阶段仅支持当前版本；仍尝试加载结构
            pass
        for raw in data.get("nodes", []):
            nid = raw["id"]
            doc._nodes[nid] = GraphNode(
                id=nid,
                type=raw["type"],
                x=float(raw.get("x", 0)),
                y=float(raw.get("y", 0)),
                params=dict(raw.get("params", {})),
            )
        for raw in data.get("edges", []):
            eid = raw["id"]
            doc._edges[eid] = GraphEdge(
                id=eid,
                src_id=raw["src_id"],
                dst_id=raw["dst_id"],
                src_port=raw.get("src_port", "out"),
                dst_port=raw.get("dst_port", "in"),
            )
        return doc

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> GraphDocument:
        return cls.from_dict(json.loads(text))
