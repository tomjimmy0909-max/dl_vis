"""形状推导：线性链（一档）与 DAG+汇合点（Add/Concat/Multiply）。

本模块负责根据计算图的拓扑关系推导每个节点输出的 NCHW 形状。
支持两种模式：
  - infer_shapes_linear_nchw：线性链，无分支/汇合
  - infer_shapes_dag_nchw：DAG，支持 Add/Concat/Multiply 汇合
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dl_vis.logic.topo_sort import topological_sort
from dl_vis.model.graph_document import GraphNode
from dl_vis.model.node_types import NodeType

if TYPE_CHECKING:
    from dl_vis.model.graph_document import GraphDocument


@dataclass
class ShapeResult:
    """形状推导结果封装。

    - ok: 推导是否成功
    - message: 总结性文字说明
    - shapes_by_node: 每节点 NCHW 形状字典，键为节点 ID
    - warnings: 推导过程中的警告信息（如参数不匹配但未阻断）
    """
    ok: bool
    message: str
    shapes_by_node: dict[str, tuple[int, int, int, int]] | None = None  # N,C,H,W
    warnings: list[str] = field(default_factory=list)


def _short_id(nid: str) -> str:
    """截断节点 ID 用于错误信息显示，避免 UUID 过长。"""
    return f"{nid[:8]}…" if len(nid) > 8 else nid


def _axis_nchw(concat_dim: int) -> int:
    """将 Concat 的拼接维（可能为负数）映射到 NCHW 的 0~3 索引。"""
    d = concat_dim + 4 if concat_dim < 0 else concat_dim
    if d not in (0, 1, 2, 3):
        raise ValueError(f"concat_dim={concat_dim} 无法映射到 NCHW 四维。")
    return d


def _propagate_unary(
    child: GraphNode,
    nc: int,
    cc: int,
    hc: int,
    wc: int,
    nxt_id: str,
    warnings: list[str],
) -> tuple[int, int, int, int] | str:
    """单输入算子：已知上游 NCHW，返回下游形状或错误文案。

    根据算子类型执行不同的形状传播规则：
    - 透传类（ReLU/Sigmoid/Softmax/Output等）：形状不变
    - 卷积类：按公式 (H + 2P - K) / S + 1 计算新尺寸
    - 池化类：与卷积类似但通道数不变
    - FC 类：展平到 (N, out_features, 1, 1)
    """
    t = child.type
    # === 透传类：形状不变 ===
    if t in (
        NodeType.RELU.value,
        NodeType.SIGMOID.value,
        NodeType.SOFTMAX.value,
        NodeType.OUTPUT.value,
        NodeType.HIST_EQUALIZE.value,
        NodeType.RESIDUAL.value,
        NodeType.PRUNE.value,
        NodeType.ATTENTION.value,
    ):
        return (nc, cc, hc, wc)
    if t == NodeType.MEL_SPECTROGRAM.value:
        p = child.params
        nm = int(p.get("n_mels", 64))
        mw = int(p.get("mel_width", 224))
        return (nc, 1, nm, mw)
    if t == NodeType.VIDEO_FRAME_PACK.value:
        p = child.params
        tf = int(p.get("max_frames", 8))
        oh = int(p.get("out_height", 224))
        ow = int(p.get("out_width", 224))
        return (nc, 3 * max(1, tf), oh, ow)
    # === BN 层：通道数校验，形状不变 ===
    if t == NodeType.BN.value:
        nf = int(child.params.get("num_features", cc))
        if nf != cc:
            warnings.append(
                f"BN 节点（{_short_id(nxt_id)}）：num_features={nf} 与上游通道 {cc} 不一致（占位警告）。"
            )
        return (nc, cc, hc, wc)
    # === 卷积层（3×3 或 1×1）：改变通道数和空间尺寸 ===
    if t == NodeType.CONV3X3.value or t == NodeType.CONV1X1.value:
        cp = child.params
        oc = int(cp.get("out_channels", cc))
        stride = int(cp.get("stride", 1))
        pad = int(cp.get("padding", 0))
        kh = 3 if t == NodeType.CONV3X3.value else 1
        kw = kh
        # 标准卷积输出尺寸公式
        nh = (hc + 2 * pad - kh) // stride + 1
        nw = (wc + 2 * pad - kw) // stride + 1
        return (nc, oc, nh, nw)
    # === 池化层：空间尺寸减半（典型），通道数不变 ===
    if t == NodeType.MAX_POOL.value or t == NodeType.AVG_POOL.value:
        cp = child.params
        ks = int(cp.get("kernel_size", 2))
        stride = int(cp.get("stride", ks))
        pad = int(cp.get("padding", 0))
        nh = (hc + 2 * pad - ks) // stride + 1
        nw = (wc + 2 * pad - ks) // stride + 1
        return (nc, cc, nh, nw)
    # === 全连接层：展平时空维，输出 (N, out_features, 1, 1) ===
    if t == NodeType.FC.value:
        flat = cc * hc * wc
        inf = int(child.params.get("in_features", flat))
        outf = int(child.params.get("out_features", 10))
        if inf != flat:
            warnings.append(
                f"FC 节点（{_short_id(nxt_id)}）：in_features={inf} 与上游展平维 C×H×W={flat} 不一致（占位校验）。"
            )
        return (nc, outf, 1, 1)
    if t == NodeType.INPUT.value:
        return "非法拓扑：Input 不应有入边。"
    return f"未知或未建模算子类型: {t}"


def _fuse_elementwise(
    shapes: list[tuple[int, int, int, int]], op_label: str
) -> tuple[int, int, int, int] | str:
    """逐元素融合（Add/Multiply）：所有输入 NCHW 必须完全一致。"""
    base = shapes[0]
    for s in shapes[1:]:
        if s != base:
            return f"{op_label}：各路输入 NCHW 须完全一致，发现 {base} 与 {s}。"
    return base


def _fuse_concat(
    shapes: list[tuple[int, int, int, int]], axis: int
) -> tuple[int, int, int, int] | str:
    """Concat 融合：沿指定维拼接，其他维必须一致。

    axis 为 NCHW 索引（0=N, 1=C, 2=H, 3=W），拼接后对应维求和。
    """
    ns = [s[0] for s in shapes]
    cs = [s[1] for s in shapes]
    hs = [s[2] for s in shapes]
    ws = [s[3] for s in shapes]
    if axis == 0:
        if not all((cs[i], hs[i], ws[i]) == (cs[0], hs[0], ws[0]) for i in range(len(shapes))):
            return "Concat(dim=N)：除批次维外 C/H/W 须一致。"
        return (sum(ns), cs[0], hs[0], ws[0])
    if axis == 1:
        if not all((ns[i], hs[i], ws[i]) == (ns[0], hs[0], ws[0]) for i in range(len(shapes))):
            return "Concat(dim=C)：除通道维外 N/H/W 须一致。"
        return (ns[0], sum(cs), hs[0], ws[0])
    if axis == 2:
        if not all((ns[i], cs[i], ws[i]) == (ns[0], cs[0], ws[0]) for i in range(len(shapes))):
            return "Concat(dim=H)：除 H 维外 N/C/W 须一致。"
        return (ns[0], cs[0], sum(hs), ws[0])
    if axis == 3:
        if not all((ns[i], cs[i], hs[i]) == (ns[0], cs[0], hs[0]) for i in range(len(shapes))):
            return "Concat(dim=W)：除 W 维外 N/C/H 须一致。"
        return (ns[0], cs[0], hs[0], sum(ws))
    return "Concat：无效拼接轴。"


def infer_shapes_dag_nchw(doc: GraphDocument) -> ShapeResult:
    """
    单 Input、无环 DAG；支持 Add / Multiply（同形）与 Concat（concat_dim）。
    「训练数据集」节点不参与张量 shape；其余计算子图须从唯一 Input 全可达。
    """
    inputs = [n for n in doc.iter_nodes() if n.type == NodeType.INPUT.value]
    if len(inputs) != 1:
        return ShapeResult(
            ok=False,
            message="DAG 推导要求图中恰好一个 Input 节点。",
            shapes_by_node=None,
        )
    root = inputs[0]
    order = topological_sort(doc)
    if order is None:
        return ShapeResult(ok=False, message="图中存在环，无法推导。", shapes_by_node=None)

    preds: dict[str, list[str]] = {nid: [] for nid in doc.nodes}
    for e in doc.edges.values():
        preds[e.dst_id].append(e.src_id)

    preds_compute: dict[str, list[str]] = {
        nid: [
            pid
            for pid in pl
            if (pnode := doc.get_node(pid)) is not None and pnode.type != NodeType.DATASET.value
        ]
        for nid, pl in preds.items()
    }

    indeg: dict[str, int] = {nid: 0 for nid in doc.nodes}
    for e in doc.edges.values():
        indeg[e.dst_id] += 1

    reach: set[str] = set()
    dq: deque[str] = deque([root.id])
    while dq:
        u = dq.popleft()
        if u in reach:
            continue
        reach.add(u)
        for e in doc.edges.values():
            if e.src_id == u and e.dst_id not in reach:
                dq.append(e.dst_id)
    compute_nodes = {nid for nid, n in doc.nodes.items() if n.type != NodeType.DATASET.value}
    if not compute_nodes.issubset(reach):
        return ShapeResult(
            ok=False,
            message="存在无法从 Input 经有向边到达的计算节点（不含数据集素材节点），请删除或改连线。",
            shapes_by_node=None,
        )

    for nid in doc.nodes:
        cn = doc.get_node(nid)
        if cn is None or cn.type == NodeType.DATASET.value or nid == root.id:
            continue
        if indeg[nid] == 0:
            return ShapeResult(
                ok=False,
                message=f"节点 {_short_id(nid)} 入度为 0 但不是 Input，DAG 不合法。",
                shapes_by_node=None,
            )

    warnings: list[str] = []
    shapes: dict[str, tuple[int, int, int, int]] = {}
    p = root.params
    n, c, h, w = (
        int(p.get("batch", 1)),
        int(p.get("channels", 3)),
        int(p.get("height", 224)),
        int(p.get("width", 224)),
    )
    shapes[root.id] = (n, c, h, w)

    for nid in order:
        if nid == root.id:
            continue
        child = doc.get_node(nid)
        if child is None:
            return ShapeResult(ok=False, message=f"缺失节点 {_short_id(nid)}。", shapes_by_node=None)
        if child.type == NodeType.DATASET.value:
            continue
        pl = preds_compute.get(nid, [])
        t = child.type

        if t == NodeType.INPUT.value:
            return ShapeResult(ok=False, message="非法拓扑：出现第二个 Input。", shapes_by_node=None)

        if len(pl) == 0:
            return ShapeResult(
                ok=False,
                message=f"节点 {_short_id(nid)}（{t}）缺少输入边。",
                shapes_by_node=None,
            )

        if len(pl) == 1:
            if t in (NodeType.ADD.value, NodeType.CONCAT.value, NodeType.MULTIPLY.value):
                return ShapeResult(
                    ok=False,
                    message=f"汇合节点 {t}（{_short_id(nid)}）至少需要两路输入。",
                    shapes_by_node=None,
                )
            ps = shapes.get(pl[0])
            if ps is None:
                return ShapeResult(
                    ok=False,
                    message=f"无法得到节点 {_short_id(pl[0])} 的输出形状（拓扑或数据异常）。",
                    shapes_by_node=None,
                )
            nc, cc, hc, wc = ps
            out = _propagate_unary(child, nc, cc, hc, wc, nid, warnings)
            if isinstance(out, str):
                return ShapeResult(ok=False, message=out, shapes_by_node=None)
            shapes[nid] = out
        else:
            if t not in (NodeType.ADD.value, NodeType.CONCAT.value, NodeType.MULTIPLY.value):
                return ShapeResult(
                    ok=False,
                    message=(
                        f"节点 {_short_id(nid)}（{t}）有多条入边，当前仅支持 Add / Concat / Multiply 作为汇合。"
                    ),
                    shapes_by_node=None,
                )
            in_shapes: list[tuple[int, int, int, int]] = []
            for pid in pl:
                sh = shapes.get(pid)
                if sh is None:
                    return ShapeResult(
                        ok=False,
                        message=f"无法得到节点 {_short_id(pid)} 的输出形状。",
                        shapes_by_node=None,
                    )
                in_shapes.append(sh)
            if t == NodeType.ADD.value:
                fused = _fuse_elementwise(in_shapes, "Add")
            elif t == NodeType.MULTIPLY.value:
                fused = _fuse_elementwise(in_shapes, "Multiply")
            else:
                try:
                    ax = _axis_nchw(int(child.params.get("concat_dim", 1)))
                except ValueError as e:
                    return ShapeResult(ok=False, message=str(e), shapes_by_node=None)
                fused = _fuse_concat(in_shapes, ax)

            if isinstance(fused, str):
                return ShapeResult(
                    ok=False,
                    message=f"节点 {_short_id(nid)}：{fused}",
                    shapes_by_node=None,
                )
            shapes[nid] = fused

    msg = (
        "【DAG】NCHW 推导完成（支持分叉与 Add/Concat/Multiply 汇合）。\n"
        "注意：菜单「导出/试跑前向/Sequential 训练」仍为线性链能力；DAG 需自定义 Module 或后续代码生成。"
    )
    return ShapeResult(ok=True, message=msg, shapes_by_node=shapes, warnings=warnings)


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
    order = topological_sort(doc)
    if order is None:
        return ShapeResult(ok=False, message="图中存在环，无法推导。", shapes_by_node=None)
    _ = order

    compute_ids = {nid for nid, n in doc.nodes.items() if n.type != NodeType.DATASET.value}
    indeg: dict[str, int] = {nid: 0 for nid in compute_ids}
    outdeg: dict[str, int] = {nid: 0 for nid in compute_ids}
    succ: dict[str, str | None] = {nid: None for nid in compute_ids}
    pred: dict[str, str | None] = {nid: None for nid in compute_ids}
    for e in doc.edges.values():
        if e.src_id not in compute_ids or e.dst_id not in compute_ids:
            continue
        indeg[e.dst_id] += 1
        outdeg[e.src_id] += 1
        if succ[e.src_id] is not None:
            return ShapeResult(
                ok=False,
                message=(
                    "存在分支（一个节点有多条出边）。\n"
                    "【一档】线性链推导不适用；请使用「DAG 形状推导」或使用 Add/Concat 汇合。"
                ),
                shapes_by_node=None,
            )
        succ[e.src_id] = e.dst_id
        if pred[e.dst_id] is not None:
            return ShapeResult(
                ok=False,
                message=(
                    "存在合并（一个节点有多条入边）。\n"
                    "【一档】线性链推导不适用；请使用 DAG 推导（Add/Concat/Multiply）。"
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
        out = _propagate_unary(child, nc, cc, hc, wc, nxt, warnings)
        if isinstance(out, str):
            return ShapeResult(ok=False, message=out, shapes_by_node=None)
        shapes[nxt] = out
        cur_id = nxt

    msg = (
        "【一档】线性链 NCHW 推导完成（单 Input、无分支/汇合）。\n"
        "含分叉/汇合请使用工具推导（infer_shapes_dag_nchw）。"
    )
    return ShapeResult(ok=True, message=msg, shapes_by_node=shapes, warnings=warnings)


def export_sequential_prerequisite_message(doc: GraphDocument) -> str | None:
    """
    导出 ``nn.Sequential`` 前校验：先 DAG 形状与连通性，再要求线性链。
    返回 ``None`` 表示可导出；否则为中文错误说明。
    """
    dag = infer_shapes_dag_nchw(doc)
    if not dag.ok:
        return "图未通过完整检查（单 Input、无环、全图从 Input 可达、汇合形状正确）：\n" + dag.message
    from dl_vis.logic.export_torch import linear_chain_order

    try:
        linear_chain_order(doc)
    except ValueError as e:
        return (
            "当前「导出 Sequential」仅支持单一路径、无分叉/汇合的线性链。\n"
            "可先保存 JSON，或使用后续版本的自定义 forward 导出。\n\n"
            f"详细原因：{e}"
        )
    return None
