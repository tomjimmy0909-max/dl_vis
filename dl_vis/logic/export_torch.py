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


def export_to_torch_module(document: GraphDocument, *, skip_trailing_softmax: bool = False) -> Any:
    """尝试构建内存中的 nn.Sequential（仅线性链）。"""
    import torch.nn as nn

    src = export_sequential_source(document, module_var="seq", skip_trailing_softmax=skip_trailing_softmax)
    g: dict[str, Any] = {}
    exec(src, {"nn": nn, "torch": __import__("torch")}, g)
    return g["seq"]


def linear_chain_order(doc: GraphDocument) -> list[str]:
    """验证「计算子图」为一条覆盖全部非 Dataset 节点的有向路径；返回该序列。

    条件：
    1. 所有计算节点构成一条单一路径（无分支、无汇合）
    2. 恰好一个入度为 0 的起点（通常为 Input）
    3. 无环、无孤立节点
    4. 边数恰好为 节点数 - 1
    """
    # 排除 Dataset 节点，仅考虑计算节点
    ds = {nid for nid, n in doc.nodes.items() if n.type == NodeType.DATASET.value}
    nodes = set(doc.nodes.keys()) - ds
    if len(nodes) == 0:
        raise ValueError("图为空（或仅含数据集节点），无法导出。")
    indeg: dict[str, int] = defaultdict(int)
    # succ/pred 记录每个节点的唯一后继/前驱（线性链要求每个节点最多一个）
    succ: dict[str, str | None] = {nid: None for nid in nodes}
    pred: dict[str, str | None] = {nid: None for nid in nodes}
    for e in doc.edges.values():
        if e.src_id in ds or e.dst_id in ds:
            continue
        if e.src_id not in nodes or e.dst_id not in nodes:
            raise ValueError("边引用了不存在的节点。")
        indeg[e.dst_id] += 1
        if succ[e.src_id] is not None:
            raise ValueError("存在分支（一个节点有多条出边），Sequential 无法表达。")
        succ[e.src_id] = e.dst_id
        if pred[e.dst_id] is not None:
            raise ValueError("存在合并（一个节点有多条入边），Sequential 无法表达。")
        pred[e.dst_id] = e.src_id

    # 找到起点（入度为 0 的唯一节点）
    roots = [nid for nid in nodes if indeg[nid] == 0]
    if len(roots) != 1:
        raise ValueError("线性链要求恰好一个入度为 0 的起点（通常为 Input）。")
    cur = roots[0]
    # 沿 succ 链遍历，构造拓扑序列
    order: list[str] = []
    seen: set[str] = set()
    while cur is not None:
        if cur in seen:
            raise ValueError("检测到环。")
        seen.add(cur)
        order.append(cur)
        cur = succ[cur]
    # 校验：必须覆盖所有计算节点
    if len(order) != len(nodes):
        raise ValueError("图不是覆盖全部计算节点的单一路径（可能存在孤立点或多分量）。")
    # 校验：计算边数必须等于 节点数 - 1（路径长度）
    n_compute_edges = sum(
        1
        for e in doc.edges.values()
        if e.src_id not in ds and e.dst_id not in ds
    )
    if n_compute_edges != max(0, len(order) - 1):
        raise ValueError("边数与线性路径不一致。")
    return order


def _repr_bool(b: bool) -> str:
    return "True" if b else "False"


def _only_output_nodes_follow(doc: GraphDocument, order: list[str], index: int) -> bool:
    """从 ``order[index]`` 之后是否仅剩 Output 类型节点（用于跳过链尾 Softmax）。"""
    for nid in order[index + 1 :]:
        n = doc.get_node(nid)
        if n is None:
            return False
        if n.type != NodeType.OUTPUT.value:
            return False
    return True


def export_sequential_source(
    doc: GraphDocument, *, module_var: str = "net", skip_trailing_softmax: bool = False
) -> str:
    """生成可执行的 Python 源码字符串：``import torch.nn`` + ``module_var = nn.Sequential(...)``。"""
    order = linear_chain_order(doc)
    layers: list[str] = []  # 收集每层对应的 nn.* 代码行
    spatial = True  # 标记当前是否在卷积特征图空间上（若为 True 且遇到 FC，需先插入 Flatten）

    for i, nid in enumerate(order):
        n = doc.get_node(nid)
        if n is None:
            raise ValueError("节点缺失。")
        t = n.type
        p = copy.deepcopy(n.params)

        # 跳过非计算节点：Input/Output/Dataset 不生成层代码
        if t == NodeType.INPUT.value:
            continue
        if t == NodeType.OUTPUT.value:
            continue
        if t == NodeType.DATASET.value:
            continue
        if t in (
            NodeType.HIST_EQUALIZE.value,
            NodeType.MEL_SPECTROGRAM.value,
            NodeType.VIDEO_FRAME_PACK.value,
        ):
            continue
        # 不支持导出为 Sequential 的复杂节点
        if t in (
            NodeType.RESIDUAL.value,
            NodeType.PRUNE.value,
            NodeType.ATTENTION.value,
            NodeType.ADD.value,
            NodeType.CONCAT.value,
            NodeType.MULTIPLY.value,
        ):
            raise ValueError(f"节点类型「{t}」当前不支持导出为 Sequential。")

        # 可选：跳过链尾紧邻 Output 的 Softmax（避免与 CrossEntropyLoss 冲突）
        if (
            skip_trailing_softmax
            and t == NodeType.SOFTMAX.value
            and _only_output_nodes_follow(doc, order, i)
        ):
            continue

        # 根据节点类型生成不同的 nn.* 代码
        if t == NodeType.CONV3X3.value:
            # Conv2d(kernel_size=3)
            ic = int(p["in_channels"])
            oc = int(p["out_channels"])
            s = int(p.get("stride", 1))
            pad = int(p.get("padding", 0))
            bias = bool(p.get("bias", True))
            layers.append(f"nn.Conv2d({ic}, {oc}, kernel_size=3, stride={s}, padding={pad}, bias={_repr_bool(bias)})")
            spatial = True
        elif t == NodeType.CONV1X1.value:
            # Conv2d(kernel_size=1)
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
            # 若前面是卷积/池化（spatial=True），先 Flatten 再 Linear
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

    # 拼接所有层代码，生成完整的 Sequential 定义
    joined = ",\n    ".join(layers)
    return (
        "import torch\n"
        "import torch.nn as nn\n\n"
        f"{module_var} = nn.Sequential(\n    {joined},\n)\n"
    )


def _effective_k_doc(doc: GraphDocument) -> int:
    """与 ``runtime_torch.effective_num_classes`` 一致（避免 export↔runtime 循环导入）。"""
    order = linear_chain_order(doc)
    last: int | None = None
    for nid in order:
        n = doc.get_node(nid)
        if n and n.type == NodeType.FC.value:
            last = int(n.params.get("out_features", 0))
    if last is None:
        raise ValueError("图中需要 FC 节点以确定分类类别数 K。")
    for o in doc.iter_nodes():
        if o.type == NodeType.OUTPUT.value:
            raw = int(o.params.get("num_classes", 0))
            if raw > 0 and raw != last:
                raise ValueError(
                    f"Output.num_classes={raw} 与最后一个 FC.out_features={last} 不一致。"
                )
            break
    return last


def _input_nchw_doc(doc: GraphDocument) -> tuple[int, int, int, int]:
    for n in doc.iter_nodes():
        if n.type == NodeType.INPUT.value:
            p = n.params
            return (
                int(p.get("batch", 1)),
                int(p.get("channels", 3)),
                int(p.get("height", 224)),
                int(p.get("width", 224)),
            )
    raise ValueError("未找到 Input 节点。")


def _output_train_meta(doc: GraphDocument) -> tuple[int, float]:
    """与输出头节点上 train_epochs / train_lr 对齐（导出与 argparse 默认值）。"""
    for n in doc.iter_nodes():
        if n.type == NodeType.OUTPUT.value:
            p = n.params
            ep = int(p.get("train_epochs", 20))
            lr = float(p.get("train_lr", 1e-3))
            return max(1, ep), lr
    return 20, 1e-3


def export_full_training_script(
    doc: GraphDocument, *, module_var: str = "net"
) -> str:
    """
    导出模型 + 与画布 Dataset→Input 对齐的训练脚手架（可单独 ``python`` 运行）。
    无可用图上数据时仅追加说明注释。
    """
    from dl_vis.logic.graph_dataset import parse_graph_linked_training

    spec = parse_graph_linked_training(doc)
    skip_softmax = bool(spec)
    head = export_sequential_source(
        doc, module_var=module_var, skip_trailing_softmax=skip_softmax
    )
    if spec is None:
        return (
            head
            + "\n\n"
            + "# [dl_vis] 未解析到可训练数据：请添加「Dataset→Input」"
            + "（文件夹=ImageFolder，单文件=.csv，或两个 .npy=特征与标签）。\n"
        )

    n_bat, c, h, w = _input_nchw_doc(doc)
    k = _effective_k_doc(doc)
    epochs_def, lr_def = _output_train_meta(doc)
    return head + "\n" + _format_training_epilogue(
        spec, module_var, n_bat, c, h, w, k, epochs_def, lr_def
    )


def _format_training_epilogue(
    spec: object,
    module_var: str,
    batch: int,
    c: int,
    h: int,
    w: int,
    k_v: int,
    epochs_def: int,
    lr_def: float,
) -> str:
    """生成可独立运行的训练段：argparse + DataLoader 批训练 + 与 Input/Output 对齐的超参。"""
    p1 = repr(spec.primary)
    p2 = repr(spec.secondary or "")
    csv_skip_lit = "True" if spec.csv_skip_header else "False"
    mode_lit = repr(spec.mode)
    lr_repr = repr(float(lr_def))

    lines: list[str] = []
    ln = lines.append

    ln('"""')
    ln("dl_vis 导出：自包含分类训练脚本（与画布节点一致，安装依赖后可直接运行）。")
    ln("  pip install torch numpy")
    ln("  图像文件夹另需: pip install torchvision")
    ln("")
    ln(f"- Input: batch={int(batch)}, C/H/W={int(c)}/{int(h)}/{int(w)}（与 DataLoader batch、Resize 对齐）")
    ln(f"- Output: NUM_CLASSES={int(k_v)}；默认 epochs={int(epochs_def)}, lr={lr_def!r}（见 DEFAULT_*，可用命令行覆盖）")
    ln(f"- 数据: Dataset→Input，模式 {spec.mode!r}")
    ln("- 损失 CrossEntropyLoss + Adam（与 dl_vis 内训练一致；链尾 Softmax 已在 Sequential 中省略）")
    ln('"""')
    ln("")
    ln("from __future__ import annotations")
    ln("")
    ln("import argparse")
    ln("import random")
    ln("import sys")
    ln("")
    ln("import numpy as np")
    ln("import torch")
    ln("import torch.nn as nn")
    ln("from torch.utils.data import DataLoader, TensorDataset")
    ln("")
    ln(f"NUM_CLASSES = {int(k_v)}")
    ln("INPUT_BATCH = " + str(int(batch)))
    ln("INPUT_C = " + str(int(c)))
    ln("INPUT_H = " + str(int(h)))
    ln("INPUT_W = " + str(int(w)))
    ln("DEFAULT_EPOCHS = " + str(int(epochs_def)))
    ln("DEFAULT_LR = " + lr_repr)
    ln("SEED = 42")
    ln(f"DLVIS_DATA_MODE = {mode_lit}")
    ln(f"DATA_PATH_PRIMARY = {p1}")
    ln(f"DATA_PATH_SECONDARY = {p2}")
    ln(f"CSV_SKIP_HEADER = {csv_skip_lit}")
    ln("")
    ln("")
    ln(f"def _model() -> nn.Sequential:")
    ln(f"    return {module_var}")
    ln("")
    ln("")
    ln("def set_seed(seed: int = SEED) -> None:")
    ln("    random.seed(seed)")
    ln("    np.random.seed(seed)")
    ln("    torch.manual_seed(seed)")
    ln("    if torch.cuda.is_available():")
    ln("        torch.cuda.manual_seed_all(seed)")
    ln("")
    ln("")
    ln("def build_train_dataloader() -> DataLoader:")
    ln("    bs = max(1, int(INPUT_BATCH))")
    ln("    if DLVIS_DATA_MODE == \"image_folder\":")
    ln("        try:")
    ln("            from torchvision import datasets, transforms")
    ln("        except ImportError:")
    ln("            print(\"缺少 torchvision，请执行: pip install torchvision\", file=sys.stderr)")
    ln("            raise SystemExit(1)")
    ln("        tfms = [transforms.Resize((INPUT_H, INPUT_W))]")
    ln("        if INPUT_C == 1:")
    ln("            tfms.append(transforms.Grayscale(num_output_channels=1))")
    ln("        elif INPUT_C != 3:")
    ln("            raise SystemExit(\"Input.channels 仅支持 1 或 3，请在画布修改输入层\")")
    ln("        tfms.append(transforms.ToTensor())")
    ln("        ds = datasets.ImageFolder(DATA_PATH_PRIMARY, transform=transforms.Compose(tfms))")
    ln("        if len(ds.classes) != NUM_CLASSES:")
    ln("            raise SystemExit(")
    ln("                f\"ImageFolder 子文件夹数 {len(ds.classes)} 与 NUM_CLASSES={NUM_CLASSES} 不一致\\n\"")
    ln("                \"（请保证 Output/FC 类别数与数据目录结构一致）\"")
    ln("            )")
    ln("        return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)")
    ln("    if DLVIS_DATA_MODE == \"csv\":")
    ln("        import csv")
    ln("        need = INPUT_C * INPUT_H * INPUT_W")
    ln("        xs: list[list[float]] = []")
    ln("        ys: list[int] = []")
    ln("        with open(DATA_PATH_PRIMARY, newline=\"\", encoding=\"utf-8-sig\") as f:")
    ln("            for i, row in enumerate(csv.reader(f)):")
    ln("                if not row:")
    ln("                    continue")
    ln("                if CSV_SKIP_HEADER and i == 0:")
    ln("                    continue")
    ln("                if len(row) < need + 1:")
    ln("                    raise SystemExit(f\"CSV 第 {i + 1} 行列数不足（需 C*H*W+1 列）\")")
    ln("                xs.append([float(row[j]) for j in range(need)])")
    ln("                ys.append(int(float(row[need]])))")
    ln("        if not xs:")
    ln("            raise SystemExit(\"CSV 无有效数据\")")
    ln("        x = torch.from_numpy(np.array(xs, np.float32).reshape(-1, INPUT_C, INPUT_H, INPUT_W))")
    ln("        y = torch.from_numpy(np.array(ys, np.int64))")
    ln("        if int(y.max()) >= NUM_CLASSES or int(y.min()) < 0:")
    ln("            raise SystemExit(\"标签须在 [0, NUM_CLASSES-1]，与画布 Output/FC 一致\")")
    ln("        return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True, num_workers=0, drop_last=False)")
    ln("    if DLVIS_DATA_MODE == \"npy_pair\":")
    ln("        x = torch.from_numpy(np.load(DATA_PATH_PRIMARY).astype(np.float32))")
    ln("        y = torch.from_numpy(np.load(DATA_PATH_SECONDARY).astype(np.int64))")
    ln("        if x.ndim != 4:")
    ln("            raise SystemExit(\"X.npy 须为 NCHW\")")
    ln("        if y.ndim != 1 or x.shape[0] != y.shape[0]:")
    ln("            raise SystemExit(\"y.npy 须为 (N,) 且与 X 批次维一致\")")
    ln("        if int(y.max()) >= NUM_CLASSES or int(y.min()) < 0:")
    ln("            raise SystemExit(\"标签须在 [0, NUM_CLASSES-1]\")")
    ln("        return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True, num_workers=0, drop_last=False)")
    ln("    raise SystemExit(f\"未知 DLVIS_DATA_MODE={DLVIS_DATA_MODE!r}\")")
    ln("")
    ln("")
    ln("def train(*, epochs: int, lr: float, device: str | None) -> None:")
    ln("    set_seed(SEED)")
    ln("    dev = torch.device(device or (\"cuda\" if torch.cuda.is_available() else \"cpu\"))")
    ln("    model = _model().to(dev)")
    ln("    loader = build_train_dataloader()")
    ln("    opt = torch.optim.Adam(model.parameters(), lr=lr)")
    ln("    crit = nn.CrossEntropyLoss()")
    ln("    model.train()")
    ln("    for ep in range(max(1, int(epochs))):")
    ln("        total = 0.0")
    ln("        nb = 0")
    ln("        for xb, yb in loader:")
    ln("            xb = xb.to(dev, non_blocking=False)")
    ln("            yb = yb.to(dev, non_blocking=False)")
    ln("            opt.zero_grad(set_to_none=True)")
    ln("            logits = model(xb)")
    ln("            if logits.dim() == 4:")
    ln("                logits = logits.view(logits.size(0), -1)")
    ln("            loss = crit(logits, yb)")
    ln("            loss.backward()")
    ln("            opt.step()")
    ln("            total += float(loss.detach().item())")
    ln("            nb += 1")
    ln("        avg = total / max(1, nb)")
    ln("        print(f\"epoch {ep + 1}/{epochs}: loss={avg:.6f} (batches={nb}, device={dev})\")")
    ln("")
    ln("")
    ln("def main() -> None:")
    ln("    p = argparse.ArgumentParser(description=\"dl_vis 导出训练（与画布一致）\")")
    ln("    p.add_argument(\"--epochs\", type=int, default=DEFAULT_EPOCHS, help=\"训练轮数（默认来自输出头 train_epochs）\")")
    ln("    p.add_argument(\"--lr\", type=float, default=DEFAULT_LR, help=\"学习率（默认来自输出头 train_lr）\")")
    ln("    p.add_argument(\"--device\", type=str, default=\"\", help=\"例如 cuda 或 cpu，空则自动\")")
    ln("    args = p.parse_args()")
    ln("    train(epochs=args.epochs, lr=float(args.lr), device=(args.device or None))")
    ln("")
    ln("")
    ln("if __name__ == \"__main__\":")
    ln("    main()")

    return "\n".join(lines) + "\n"
