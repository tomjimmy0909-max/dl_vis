"""从右侧「代码」面板解析用户编辑的 PyTorch 片段，写回节点 params。"""

from __future__ import annotations

import ast
from typing import Any

from dl_vis.model.node_types import DEFAULT_PARAMS, NodeType

_CODE_TYPES = frozenset(
    {
        NodeType.INPUT.value,
        NodeType.CONV3X3.value,
        NodeType.CONV1X1.value,
        NodeType.MAX_POOL.value,
        NodeType.AVG_POOL.value,
        NodeType.FC.value,
        NodeType.RELU.value,
        NodeType.SOFTMAX.value,
        NodeType.BN.value,
    }
)


def code_params_editable(node_type: str) -> bool:
    return node_type in _CODE_TYPES


def apply_code_preview_text(node_type: str, text: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    解析 ``code_preview_for_node`` 风格的文本。
    成功返回 ``(patch, None)``；失败 ``(None, 中文错误信息)``。
    """
    try:
        patch = _parse(node_type, text)
    except ValueError as e:
        return None, str(e)
    except SyntaxError as e:
        return None, f"语法无法解析：{e}"
    return patch, None


def _first_meaningful_line(text: str) -> str | None:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return None


def _int_expr(n: ast.expr, field: str) -> int:
    if isinstance(n, ast.Constant) and type(n.value) is int:
        return int(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -_int_expr(n.operand, field)
    raise ValueError(f"「{field}」须为整数常量")


def _float_expr(n: ast.expr, field: str) -> float:
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return float(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -_float_expr(n.operand, field)
    raise ValueError(f"「{field}」须为数值常量")


def _bool_expr(n: ast.expr, field: str) -> bool:
    if isinstance(n, ast.Constant) and isinstance(n.value, bool):
        return bool(n.value)
    raise ValueError(f"「{field}」须为 True / False")


def _fn_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    raise ValueError("仅支持 nn.* / torch.* 形式的调用（如 nn.Conv2d(...)）")


def _parse_assign_call(line: str) -> ast.Call:
    mod = ast.parse(line, mode="exec")
    if len(mod.body) != 1 or not isinstance(mod.body[0], ast.Assign):
        raise ValueError("Input 代码需为赋值：如 x = torch.randn(N, C, H, W)")
    asn = mod.body[0]
    if len(asn.targets) != 1 or not isinstance(asn.value, ast.Call):
        raise ValueError("Input 右侧须为 torch.randn(...)")
    return asn.value


def _parse_expr_call(line: str) -> ast.Call:
    mod = ast.parse(f"__code_x = {line}", mode="exec")
    if len(mod.body) != 1 or not isinstance(mod.body[0], ast.Assign):
        raise ValueError("无法解析该表达式")
    val = mod.body[0].value
    if not isinstance(val, ast.Call):
        raise ValueError("该行须为函数调用，如 nn.Conv2d(...)")
    return val


def _kw_map(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _parse_input(text: str) -> dict[str, Any]:
    line = _first_meaningful_line(text)
    if not line:
        raise ValueError("未找到可解析的 torch.randn 行")
    call = _parse_assign_call(line)
    if _fn_name(call) != "randn":
        raise ValueError("Input 请使用 torch.randn(N, C, H, W)")
    if len(call.args) < 4:
        raise ValueError("torch.randn 至少需要 4 个位置参数：N, C, H, W")
    return {
        "batch": _int_expr(call.args[0], "N"),
        "channels": _int_expr(call.args[1], "C"),
        "height": _int_expr(call.args[2], "H"),
        "width": _int_expr(call.args[3], "W"),
    }


def _expect_kernel(call: ast.Call, expect: int, label: str) -> None:
    kws = _kw_map(call)
    if "kernel_size" in kws:
        ks = _int_expr(kws["kernel_size"], "kernel_size")
        if ks != expect:
            raise ValueError(f"{label} 固定 kernel_size={expect}，当前为 {ks}")
    elif len(call.args) > 2:
        ks = _int_expr(call.args[2], "kernel_size")
        if ks != expect:
            raise ValueError(f"{label} 固定 kernel_size={expect}，当前为 {ks}")
    # 若省略 kernel_size，由类型隐式约定，不报错


def _parse_conv2d(text: str, expect_ks: int, label: str) -> dict[str, Any]:
    line = _first_meaningful_line(text)
    if not line:
        raise ValueError("未找到 nn.Conv2d(...) 行")
    call = _parse_expr_call(line)
    if _fn_name(call) != "Conv2d":
        raise ValueError("该行须为 nn.Conv2d(...)")
    _expect_kernel(call, expect_ks, label)
    kws = _kw_map(call)
    defaults = (
        DEFAULT_PARAMS[NodeType.CONV3X3.value]
        if expect_ks == 3
        else DEFAULT_PARAMS[NodeType.CONV1X1.value]
    )
    if len(call.args) < 2:
        raise ValueError("nn.Conv2d 至少需要 in_channels、out_channels 两个位置参数")
    ic = _int_expr(call.args[0], "in_channels")
    oc = _int_expr(call.args[1], "out_channels")
    stride = _int_expr(kws["stride"], "stride") if "stride" in kws else int(defaults["stride"])
    padding = _int_expr(kws["padding"], "padding") if "padding" in kws else int(defaults["padding"])
    bias = _bool_expr(kws["bias"], "bias") if "bias" in kws else bool(defaults["bias"])
    return {
        "in_channels": ic,
        "out_channels": oc,
        "stride": stride,
        "padding": padding,
        "bias": bias,
    }


def _parse_pool(text: str, cls: str) -> dict[str, Any]:
    line = _first_meaningful_line(text)
    if not line:
        raise ValueError(f"未找到 nn.{cls}(...) 行")
    call = _parse_expr_call(line)
    if _fn_name(call) != cls:
        raise ValueError(f"该行须为 nn.{cls}(...)")
    kws = _kw_map(call)
    dmax = DEFAULT_PARAMS[NodeType.MAX_POOL.value]
    ks: int
    if "kernel_size" in kws:
        ks = _int_expr(kws["kernel_size"], "kernel_size")
    elif call.args:
        ks = _int_expr(call.args[0], "kernel_size")
    else:
        raise ValueError("缺少 kernel_size")
    st = _int_expr(kws["stride"], "stride") if "stride" in kws else ks
    pad = _int_expr(kws["padding"], "padding") if "padding" in kws else int(dmax["padding"])
    return {"kernel_size": ks, "stride": st, "padding": pad}


def _parse_linear(text: str) -> dict[str, Any]:
    line = _first_meaningful_line(text)
    if not line:
        raise ValueError("未找到 nn.Linear(...) 行")
    call = _parse_expr_call(line)
    if _fn_name(call) != "Linear":
        raise ValueError("该行须为 nn.Linear(...)")
    kws = _kw_map(call)
    d = DEFAULT_PARAMS[NodeType.FC.value]
    if len(call.args) < 2:
        raise ValueError("nn.Linear 至少需要 in_features、out_features")
    inf = _int_expr(call.args[0], "in_features")
    outf = _int_expr(call.args[1], "out_features")
    bias = _bool_expr(kws["bias"], "bias") if "bias" in kws else bool(d["bias"])
    return {"in_features": inf, "out_features": outf, "bias": bias}


def _parse_relu(text: str) -> dict[str, Any]:
    line = _first_meaningful_line(text)
    if not line:
        raise ValueError("未找到 nn.ReLU(...) 行")
    call = _parse_expr_call(line)
    if _fn_name(call) != "ReLU":
        raise ValueError("该行须为 nn.ReLU(...)")
    kws = _kw_map(call)
    d = DEFAULT_PARAMS[NodeType.RELU.value]
    ip = _bool_expr(kws["inplace"], "inplace") if "inplace" in kws else bool(d["inplace"])
    return {"inplace": ip}


def _parse_softmax(text: str) -> dict[str, Any]:
    line = _first_meaningful_line(text)
    if not line:
        raise ValueError("未找到 nn.Softmax(...) 行")
    call = _parse_expr_call(line)
    if _fn_name(call) != "Softmax":
        raise ValueError("该行须为 nn.Softmax(...)")
    kws = _kw_map(call)
    d = DEFAULT_PARAMS[NodeType.SOFTMAX.value]
    dim = _int_expr(kws["dim"], "dim") if "dim" in kws else int(d["dim"])
    return {"dim": dim}


def _parse_bn2d(text: str) -> dict[str, Any]:
    line = _first_meaningful_line(text)
    if not line:
        raise ValueError("未找到 nn.BatchNorm2d(...) 行")
    call = _parse_expr_call(line)
    if _fn_name(call) != "BatchNorm2d":
        raise ValueError("该行须为 nn.BatchNorm2d(...)")
    kws = _kw_map(call)
    d = DEFAULT_PARAMS[NodeType.BN.value]
    if not call.args:
        raise ValueError("nn.BatchNorm2d 需要 num_features 位置参数")
    nf = _int_expr(call.args[0], "num_features")
    eps = _float_expr(kws["eps"], "eps") if "eps" in kws else float(d["eps"])
    momentum = _float_expr(kws["momentum"], "momentum") if "momentum" in kws else float(d["momentum"])
    affine = _bool_expr(kws["affine"], "affine") if "affine" in kws else bool(d["affine"])
    return {"num_features": nf, "eps": eps, "momentum": momentum, "affine": affine}


def _parse(node_type: str, text: str) -> dict[str, Any]:
    if node_type == NodeType.INPUT.value:
        return _parse_input(text)
    if node_type == NodeType.CONV3X3.value:
        return _parse_conv2d(text, 3, "Conv3x3")
    if node_type == NodeType.CONV1X1.value:
        return _parse_conv2d(text, 1, "Conv1x1")
    if node_type == NodeType.MAX_POOL.value:
        return _parse_pool(text, "MaxPool2d")
    if node_type == NodeType.AVG_POOL.value:
        return _parse_pool(text, "AvgPool2d")
    if node_type == NodeType.FC.value:
        return _parse_linear(text)
    if node_type == NodeType.RELU.value:
        return _parse_relu(text)
    if node_type == NodeType.SOFTMAX.value:
        return _parse_softmax(text)
    if node_type == NodeType.BN.value:
        return _parse_bn2d(text)
    raise ValueError(f"节点类型「{node_type}」不支持从代码写回参数")
