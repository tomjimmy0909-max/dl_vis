"""从 ``.py`` 源码解析简单 ``nn.Module``（线性定义 + 顺序 forward），生成 ``GraphDocument`` MVP。

仅支持常见写法：**``__init__`` 内** ``self.x = nn.Conv2d(...)`` / ``Linear`` 等，或 **一层** ``nn.Sequential(...)`` 展开；
**``forward``** 中为 ``x = self.a(x); x = self.b(x)`` 顺序（无 if/for/while）。
更复杂的结构解析失败时抛出 ``ValueError``（中文说明）。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dl_vis.model.graph_document import GraphDocument
from dl_vis.model.node_types import NodeType, default_params_for_type


@dataclass
class _Layer:
    """解析得到的单层（或 Sequential 子层）。"""

    hint: str
    node_type: str
    params: dict[str, Any] = field(default_factory=dict)


def _extends_nn_module(class_node: ast.ClassDef) -> bool:
    """检查 AST 类节点是否继承 nn.Module（支持 ``nn.Module`` 和 ``Module`` 两种写法）。"""
    for b in class_node.bases:
        if isinstance(b, ast.Attribute) and b.attr == "Module":
            return True
        if isinstance(b, ast.Name) and b.id == "Module":
            return True
    return False


def _int_from_ast(node: ast.expr | None) -> int | None:
    """从 AST 节点尝试提取整数值。支持负数、乘法、加法表达式。"""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _int_from_ast(node.operand)
        return -v if v is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        a = _int_from_ast(node.left)
        b = _int_from_ast(node.right)
        if a is not None and b is not None:
            return a * b
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = _int_from_ast(node.left)
        b = _int_from_ast(node.right)
        if a is not None and b is not None:
            return a + b
    return None


def _float_from_ast(node: ast.expr | None) -> float | None:
    """从 AST 节点尝试提取浮点数值。"""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _bool_from_ast(node: ast.expr | None) -> bool | None:
    """从 AST 节点尝试提取布尔值。"""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    return None


def _call_base_name(call: ast.Call) -> str | None:
    """提取函数调用名（如 nn.Conv2d → "Conv2d", torch.randn → "randn"）。"""
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _kw_dict(call: ast.Call) -> dict[str, ast.expr]:
    """提取函数调用的关键字参数字典（排除 None key）。"""
    return {k.arg: k.value for k in call.keywords if k.arg is not None}


def _merge_conv2d(call: ast.Call) -> dict[str, Any] | None:
    args = call.args
    kw = _kw_dict(call)

    def arg(i: int) -> ast.expr | None:
        return args[i] if i < len(args) else None

    in_ch = _int_from_ast(arg(0) or kw.get("in_channels"))
    out_ch = _int_from_ast(arg(1) or kw.get("out_channels"))
    ks_node = arg(2) or kw.get("kernel_size")
    stride_node = arg(3) if len(args) > 3 else kw.get("stride")
    pad_node = arg(4) if len(args) > 4 else kw.get("padding")
    bias_node = kw.get("bias")

    k: int | None = None
    if ks_node is not None:
        if isinstance(ks_node, ast.Tuple) and len(ks_node.elts) >= 2:
            a = _int_from_ast(ks_node.elts[0])
            b = _int_from_ast(ks_node.elts[1])
            if a is not None and a == b:
                k = a
        else:
            k = _int_from_ast(ks_node)

    if in_ch is None or out_ch is None or k is None:
        return None

    stride = _int_from_ast(stride_node) if stride_node is not None else 1
    padding = _int_from_ast(pad_node) if pad_node is not None else 0
    bias = bool(_bool_from_ast(bias_node)) if bias_node is not None else True

    node_type = NodeType.CONV3X3.value if k == 3 else NodeType.CONV1X1.value if k == 1 else ""
    if not node_type:
        return None
    p = default_params_for_type(node_type)
    p.update(
        {
            "in_channels": in_ch,
            "out_channels": out_ch,
            "stride": stride,
            "padding": padding,
            "bias": bias,
        }
    )
    return {"node_type": node_type, "params": p}


def _merge_linear(call: ast.Call) -> dict[str, Any] | None:
    args = call.args
    kw = _kw_dict(call)
    inf = _int_from_ast(args[0] if len(args) > 0 else None) or _int_from_ast(kw.get("in_features"))
    outf = _int_from_ast(args[1] if len(args) > 1 else None) or _int_from_ast(kw.get("out_features"))
    bias_node = kw.get("bias")
    if inf is None or outf is None:
        return None
    bias = bool(_bool_from_ast(bias_node)) if bias_node is not None else True
    p = default_params_for_type(NodeType.FC.value)
    p.update({"in_features": inf, "out_features": outf, "bias": bias})
    return {"node_type": NodeType.FC.value, "params": p}


def _merge_maxpool(call: ast.Call) -> dict[str, Any] | None:
    kw = _kw_dict(call)
    args = call.args
    ks = (
        _int_from_ast(args[0] if args else None)
        or _int_from_ast(kw.get("kernel_size"))
        or 2
    )
    st = _int_from_ast(args[1] if len(args) > 1 else None) or _int_from_ast(kw.get("stride"))
    if st is None:
        st = ks
    pad = _int_from_ast(kw.get("padding")) or 0
    p = default_params_for_type(NodeType.MAX_POOL.value)
    p.update({"kernel_size": ks, "stride": st, "padding": pad})
    return {"node_type": NodeType.MAX_POOL.value, "params": p}


def _merge_avgpool(call: ast.Call) -> dict[str, Any] | None:
    kw = _kw_dict(call)
    args = call.args
    ks = (
        _int_from_ast(args[0] if args else None)
        or _int_from_ast(kw.get("kernel_size"))
        or 2
    )
    st = _int_from_ast(args[1] if len(args) > 1 else None) or _int_from_ast(kw.get("stride"))
    if st is None:
        st = ks
    pad = _int_from_ast(kw.get("padding")) or 0
    p = default_params_for_type(NodeType.AVG_POOL.value)
    p.update({"kernel_size": ks, "stride": st, "padding": pad})
    return {"node_type": NodeType.AVG_POOL.value, "params": p}


def _merge_bn2d(call: ast.Call) -> dict[str, Any] | None:
    args = call.args
    kw = _kw_dict(call)
    nf = _int_from_ast(args[0] if args else None) or _int_from_ast(kw.get("num_features"))
    if nf is None:
        return None
    eps = _float_from_ast(kw.get("eps"))
    momentum = _float_from_ast(kw.get("momentum"))
    affine_n = kw.get("affine")
    p = default_params_for_type(NodeType.BN.value)
    p["num_features"] = nf
    if eps is not None:
        p["eps"] = eps
    if momentum is not None:
        p["momentum"] = momentum
    if affine_n is not None and _bool_from_ast(affine_n) is not None:
        p["affine"] = bool(_bool_from_ast(affine_n))
    return {"node_type": NodeType.BN.value, "params": p}


def _merge_relu(call: ast.Call) -> dict[str, Any] | None:
    kw = _kw_dict(call)
    p = default_params_for_type(NodeType.RELU.value)
    if kw.get("inplace") is not None and _bool_from_ast(kw.get("inplace")) is not None:
        p["inplace"] = bool(_bool_from_ast(kw.get("inplace")))
    return {"node_type": NodeType.RELU.value, "params": p}


def _merge_softmax(call: ast.Call) -> dict[str, Any] | None:
    kw = _kw_dict(call)
    p = default_params_for_type(NodeType.SOFTMAX.value)
    d = _int_from_ast(kw.get("dim"))
    if d is not None:
        p["dim"] = d
    return {"node_type": NodeType.SOFTMAX.value, "params": p}


def _parse_nn_call(call: ast.Call, hint: str) -> list[_Layer]:
    base = _call_base_name(call)
    if base is None:
        return []
    merged: dict[str, Any] | None = None
    if base == "Conv2d":
        merged = _merge_conv2d(call)
    elif base == "Linear":
        merged = _merge_linear(call)
    elif base == "MaxPool2d":
        merged = _merge_maxpool(call)
    elif base == "AvgPool2d":
        merged = _merge_avgpool(call)
    elif base == "BatchNorm2d":
        merged = _merge_bn2d(call)
    elif base == "ReLU":
        merged = _merge_relu(call)
    elif base == "Sigmoid":
        merged = {"node_type": NodeType.SIGMOID.value, "params": default_params_for_type(NodeType.SIGMOID.value)}
    elif base == "Softmax":
        merged = _merge_softmax(call)
    elif base == "Sequential":
        out: list[_Layer] = []
        for i, arg in enumerate(call.args):
            if isinstance(arg, ast.Call):
                sub = _parse_nn_call(arg, f"{hint}[{i}]")
                out.extend(sub)
        return out

    if merged is None:
        return []
    nt = str(merged["node_type"])
    params = dict(merged["params"])
    return [_Layer(hint=hint, node_type=nt, params=params)]


def _layers_from_init(class_body: list[ast.stmt]) -> dict[str, list[_Layer]]:
    """``self.name`` -> 解析层列表（Sequential 可能对应多项）。"""
    by_sub: dict[str, list[_Layer]] = {}
    init_fn: ast.FunctionDef | None = None
    for item in class_body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_fn = item
            break
    if init_fn is None:
        return by_sub

    for stmt in init_fn.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                    name = tgt.attr
                    val = stmt.value
                    if isinstance(val, ast.Call):
                        layers = _parse_nn_call(val, name)
                        if layers:
                            by_sub[name] = layers
    return by_sub


def _forward_linear_order(fn: ast.FunctionDef) -> list[str] | None:
    order: list[str] = []
    for stmt in fn.body:
        if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            return None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            val = stmt.value
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
                if isinstance(val.func.value, ast.Name) and val.func.value.id == "self":
                    order.append(val.func.attr)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            c = stmt.value
            if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name) and c.func.value.id == "self":
                order.append(c.func.attr)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            val = stmt.value
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
                if isinstance(val.func.value, ast.Name) and val.func.value.id == "self":
                    order.append(val.func.attr)
    return order


def _build_ordered_layers(
    by_sub: dict[str, list[_Layer]], forward_order: list[str] | None
) -> list[_Layer]:
    if not by_sub:
        raise ValueError("未在 __init__ 中解析到任何 nn 层（需 self.xxx = nn.Conv2d / Linear 等）。")

    layers_flat: list[_Layer] = []
    if forward_order is not None and len(forward_order) > 0:
        used: set[str] = set()
        for name in forward_order:
            if name in by_sub:
                layers_flat.extend(by_sub[name])
                used.add(name)
        for name, ls in by_sub.items():
            if name not in used:
                layers_flat.extend(ls)
    elif forward_order is not None and len(forward_order) == 0:
        for _name, ls in by_sub.items():
            layers_flat.extend(ls)
    else:
        # forward 含分支/循环等，无法确定顺序，按 __init__ 出现顺序串联所有子模块
        for _name, ls in by_sub.items():
            layers_flat.extend(ls)

    if not layers_flat:
        raise ValueError("未得到有序层列表；请检查 forward 是否为顺序 self.layer(x) 形式。")
    return layers_flat


def graph_document_from_source(source: str) -> GraphDocument:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Python 语法错误：{e}") from e

    if not isinstance(tree, ast.Module):
        raise ValueError("根节点不是 ast.Module。")

    module_classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and _extends_nn_module(n)]
    if not module_classes:
        raise ValueError("未找到继承 nn.Module 的类（基类需为 nn.Module 或 Module）。")

    class_node = module_classes[0]
    by_sub = _layers_from_init(class_node.body)
    forward_fn: ast.FunctionDef | None = None
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            forward_fn = item
            break

    fo: list[str] | None = None
    if forward_fn is not None:
        fo = _forward_linear_order(forward_fn)

    ordered = _build_ordered_layers(by_sub, fo)

    doc = GraphDocument()
    dx = 190.0
    x, y = 0.0, 0.0

    in_channels = 3
    first = ordered[0]
    if first.node_type in (NodeType.CONV3X3.value, NodeType.CONV1X1.value, NodeType.BN.value):
        in_channels = int(first.params.get("in_channels", first.params.get("num_features", 3)))

    inp = doc.add_node(NodeType.INPUT.value, x, y, params={"batch": 1, "channels": in_channels, "height": 224, "width": 224})
    prev_id = inp.id
    x += dx

    last_fc_out: int | None = None
    for lay in ordered:
        n = doc.add_node(lay.node_type, x, y, params=lay.params)
        e, err = doc.add_edge(prev_id, n.id)
        if err or e is None:
            raise ValueError(f"无法添加边：{err}")
        prev_id = n.id
        x += dx
        if lay.node_type == NodeType.FC.value:
            last_fc_out = int(lay.params.get("out_features", 0))

    out_k = last_fc_out or 10
    outp = doc.add_node(
        NodeType.OUTPUT.value,
        x,
        y,
        params={
            "name": "logits",
            "task": "classify",
            "num_classes": out_k,
            "loss": "cross_entropy",
        },
    )
    e, err = doc.add_edge(prev_id, outp.id)
    if err or e is None:
        raise ValueError(f"无法连接 Output：{err}")

    return doc


def graph_document_from_py_file(path: Path) -> GraphDocument:
    p = path.expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"不是有效文件：{p}")
    if p.suffix.lower() != ".py":
        raise ValueError("当前仅支持 .py 文件。")
    text = p.read_text(encoding="utf-8")
    return graph_document_from_source(text)
