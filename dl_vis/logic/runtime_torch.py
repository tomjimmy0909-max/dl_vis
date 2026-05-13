"""图内 PyTorch 运行时：形状检查、构建 Sequential、前向试跑与合成训练。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from dl_vis.logic.export_torch import export_to_torch_module, linear_chain_order
from dl_vis.logic.shape_inference import ShapeResult, infer_shapes_dag_nchw
from dl_vis.model.node_types import NodeType

if TYPE_CHECKING:
    import torch

    from dl_vis.model.graph_document import GraphDocument


MSG_TORCH_MISSING = (
    "未检测到已安装的 PyTorch（import torch 失败）。\n"
    "请在本机 Python 环境中安装 torch 后再使用「试跑前向」或「合成训练」。"
)


@dataclass
class RunnableCheck:
    ok: bool
    shape: ShapeResult
    error: str | None = None


def check_graph_runnable(doc: GraphDocument) -> RunnableCheck:
    """形状推导；失败时 ``error`` 为结构化中文说明。"""
    res = infer_shapes_dag_nchw(doc)
    if not res.ok:
        return RunnableCheck(ok=False, shape=res, error=res.message)
    return RunnableCheck(ok=True, shape=res, error=None)


def build_sequential(doc: GraphDocument, *, skip_trailing_softmax: bool = False) -> Any:
    """
    构建 ``nn.Sequential``；无 torch 或线性链/导出错误时抛出 ``ImportError`` / ``ValueError``。
    """
    try:
        return export_to_torch_module(doc, skip_trailing_softmax=skip_trailing_softmax)
    except ImportError as e:
        raise ImportError(MSG_TORCH_MISSING) from e


def _input_nchw(doc: GraphDocument) -> tuple[int, int, int, int]:
    inputs = [n for n in doc.iter_nodes() if n.type == NodeType.INPUT.value]
    if len(inputs) != 1:
        raise ValueError("图中需恰好一个 Input 节点以构造虚拟张量。")
    p = inputs[0].params
    return (
        int(p.get("batch", 1)),
        int(p.get("channels", 3)),
        int(p.get("height", 224)),
        int(p.get("width", 224)),
    )


def dummy_forward(doc: GraphDocument) -> dict[str, Any]:
    """
    ``torch.no_grad()`` 下前向一次；返回 ``output_shape``、标量统计与可选 ``note``。
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(MSG_TORCH_MISSING) from e

    chk = check_graph_runnable(doc)
    if not chk.ok:
        raise ValueError(chk.error or chk.shape.message)
    n, c, h, w = _input_nchw(doc)
    model = build_sequential(doc, skip_trailing_softmax=False)
    x = torch.randn(n, c, h, w)
    with torch.no_grad():
        y = model(x)
    return {
        "output_shape": tuple(int(d) for d in y.shape),
        "mean": float(y.mean().item()),
        "std": float(y.std().item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
    }


def last_fc_out_features(doc: GraphDocument) -> int | None:
    """线性链上最后一个 FC 的 ``out_features``；若无 FC 返回 ``None``。"""
    try:
        order = linear_chain_order(doc)
    except ValueError:
        return None
    last: int | None = None
    for nid in order:
        n = doc.get_node(nid)
        if n and n.type == NodeType.FC.value:
            last = int(n.params.get("out_features", 0))
    return last


def effective_num_classes(doc: GraphDocument) -> tuple[int | None, str | None]:
    """
    根据 Output 节点 ``num_classes``（0 表示继承末层 FC）返回类别数 ``K``。
    若链上无 FC 或配置非法，返回 ``(None, 错误说明)``。
    """
    k_fc = last_fc_out_features(doc)
    if k_fc is None:
        return None, "当前仅支持以 FC 作为分类 logits 源：图中未找到 FC 节点。"

    outputs = [n for n in doc.iter_nodes() if n.type == NodeType.OUTPUT.value]
    for o in outputs:
        raw = int(o.params.get("num_classes", 0))
        if raw <= 0:
            continue
        if raw != k_fc:
            return None, f"Output 的类别数 num_classes={raw} 与链上最后一个 FC 的 out_features={k_fc} 不一致。"
    return k_fc, None


def validate_output_for_train(doc: GraphDocument) -> str | None:
    """运行/训练前校验 Output 与末层维度；无错误返回 ``None``。"""
    _k, err = effective_num_classes(doc)
    if err:
        return err
    outputs = [n for n in doc.iter_nodes() if n.type == NodeType.OUTPUT.value]
    for o in outputs:
        task = str(o.params.get("task", "classify"))
        loss = str(o.params.get("loss", "cross_entropy"))
        if task != "classify":
            return f"Output 任务类型「{task}」当前未实现，仅支持 classify。"
        if loss != "cross_entropy":
            return f"Output 损失「{loss}」当前未实现，仅支持 cross_entropy。"
    return None


def train_synthetic_ce(
    doc: GraphDocument,
    *,
    epochs: int,
    lr: float = 1e-3,
    skip_trailing_softmax: bool = True,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]:
    """
    合成随机 ``x``（与 Input 同形）与 ``y ~ Uniform(0, K-1)``，``CrossEntropyLoss`` + ``Adam``。
    链尾若存在紧邻 Output 的 Softmax，默认在模型中跳过该层以免与 CE 重复。
    """
    import torch
    import torch.nn as nn

    err = validate_output_for_train(doc)
    if err:
        raise ValueError(err)

    k, err2 = effective_num_classes(doc)
    if err2 or k is None:
        raise ValueError(err2 or "无法确定类别数 K。")

    chk = check_graph_runnable(doc)
    if not chk.ok:
        raise ValueError(chk.error or chk.shape.message)

    n, c, h, w = _input_nchw(doc)
    model = build_sequential(doc, skip_trailing_softmax=skip_trailing_softmax)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    losses: list[float] = []
    for ep in range(max(1, int(epochs))):
        x = torch.randn(n, c, h, w)
        y = torch.randint(0, k, (n,), dtype=torch.long)
        opt.zero_grad()
        logits = model(x)
        if logits.dim() == 4:
            logits = logits.view(logits.size(0), -1)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        lv = float(loss.detach().item())
        losses.append(lv)
        if on_epoch is not None:
            on_epoch(ep + 1, lv)
    return losses


def load_npy_training_pair(
    x_path: str,
    y_path: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """加载 ``.npy``：``X`` 形状 ``(N,C,H,W)``，``y`` 形状 ``(N,)`` 整型标签。"""
    import numpy as np
    import torch

    x_arr = np.load(x_path, allow_pickle=False)
    y_arr = np.load(y_path, allow_pickle=False)
    if x_arr.ndim != 4:
        raise ValueError(f"期望 X 为 4 维 NCHW，实际 ndim={x_arr.ndim}。")
    if y_arr.ndim != 1:
        raise ValueError(f"期望 y 为 1 维 (N,)，实际 shape={y_arr.shape}。")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X 与 y 的批次维 N 不一致。")
    x_t = torch.from_numpy(x_arr.astype("float32", copy=False))
    y_t = torch.from_numpy(y_arr.astype("int64", copy=False))
    return x_t, y_t


def load_csv_nchw_labels(
    path: str,
    *,
    channels: int,
    height: int,
    width: int,
    skip_header_row: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    读取 CSV：每行前 ``C*H*W`` 列为特征（按行主序展平再 reshape 为 NCHW），最后一列为整型标签。
    """
    import csv

    import numpy as np
    import torch

    need = int(channels) * int(height) * int(width)
    xs: list[list[float]] = []
    ys: list[int] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            if skip_header_row and i == 0:
                continue
            if len(row) < need + 1:
                raise ValueError(f"第 {i + 1} 行列数不足：需要至少 {need + 1} 列（特征+标签）。")
            feat = [float(row[j]) for j in range(need)]
            lab = int(float(row[need]))
            xs.append(feat)
            ys.append(lab)
    if not xs:
        raise ValueError("CSV 无有效数据行。")
    arr = np.array(xs, dtype=np.float32).reshape(len(xs), channels, height, width)
    y_arr = np.array(ys, dtype=np.int64)
    return torch.from_numpy(arr), torch.from_numpy(y_arr)


def train_with_tensors(
    doc: GraphDocument,
    x: "torch.Tensor",
    y: "torch.Tensor",
    *,
    epochs: int,
    lr: float = 1e-3,
    skip_trailing_softmax: bool = True,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]:
    """使用给定 ``x, y`` 在整图上训练（与 ``train_synthetic_ce`` 相同损失与优化器）。"""
    import torch
    import torch.nn as nn

    err = validate_output_for_train(doc)
    if err:
        raise ValueError(err)
    k, err2 = effective_num_classes(doc)
    if err2 or k is None:
        raise ValueError(err2 or "无法确定类别数 K。")
    chk = check_graph_runnable(doc)
    if not chk.ok:
        raise ValueError(chk.error or chk.shape.message)

    if int(y.max().item()) >= k or int(y.min().item()) < 0:
        raise ValueError(f"标签需在 [0, {k - 1}] 内，与当前 K={k} 一致。")

    model = build_sequential(doc, skip_trailing_softmax=skip_trailing_softmax)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses: list[float] = []
    n = x.size(0)
    for ep in range(max(1, int(epochs))):
        opt.zero_grad()
        logits = model(x)
        if logits.dim() == 4:
            logits = logits.view(logits.size(0), -1)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        lv = float(loss.detach().item())
        losses.append(lv)
        if on_epoch is not None:
            on_epoch(ep + 1, lv)
    return losses


def build_imagefolder_tv_transforms(channels: int, height: int, width: int):
    """与导出脚本一致：Resize +（C=1 时 Grayscale）+ ToTensor。"""
    try:
        from torchvision import transforms
    except ImportError as e:
        raise ImportError("加载图像文件夹需要 torchvision，请执行：pip install torchvision") from e

    tlist = [transforms.Resize((int(height), int(width)))]
    ch = int(channels)
    if ch == 1:
        tlist.append(transforms.Grayscale(num_output_channels=1))
    elif ch != 3:
        raise ValueError("Input.channels 仅支持 1（灰度）或 3（RGB），与 ImageFolder 加载一致。")
    tlist.append(transforms.ToTensor())
    return transforms.Compose(tlist)


def load_imagefolder_training_tensors(
    root: str,
    *,
    channels: int,
    height: int,
    width: int,
    expected_num_classes: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """将 ImageFolder 全部样本载入内存为 ``(N,C,H,W)`` 与 ``(N,)`` long 标签。"""
    try:
        from torchvision import datasets
    except ImportError as e:
        raise ImportError("加载图像文件夹需要 torchvision，请执行：pip install torchvision") from e

    import torch

    tfm = build_imagefolder_tv_transforms(channels, height, width)
    ds = datasets.ImageFolder(root, transform=tfm)
    if len(ds) == 0:
        raise ValueError("ImageFolder 目录下无可用样本。")
    if len(ds.classes) != int(expected_num_classes):
        raise ValueError(
            f"子文件夹类别数 {len(ds.classes)} 与图中 NUM_CLASSES={expected_num_classes} 不一致。"
        )
    xs = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)
    ys = torch.tensor([int(ds[i][1]) for i in range(len(ds))], dtype=torch.long)
    return xs, ys


def load_training_tensors_from_graph(doc: GraphDocument) -> tuple["torch.Tensor", "torch.Tensor"]:
    """根据 Dataset→Input 解析结果加载 ``x, y``；失败抛出带提示的 ``ValueError`` / ``ImportError``。"""
    from dl_vis.logic.graph_dataset import describe_graph_training_hint, parse_graph_linked_training

    spec = parse_graph_linked_training(doc)
    if spec is None:
        raise ValueError(describe_graph_training_hint(doc))
    err = validate_output_for_train(doc)
    if err:
        raise ValueError(err)
    k, kerr = effective_num_classes(doc)
    if kerr or k is None:
        raise ValueError(kerr or "无法确定类别数 K。")
    _n, c, h, w = _input_nchw(doc)

    if spec.mode == "image_folder":
        x, y = load_imagefolder_training_tensors(
            spec.primary, channels=c, height=h, width=w, expected_num_classes=k
        )
        if int(x.size(1)) != int(c):
            raise ValueError(f"图像张量通道 {x.size(1)} 与 Input.channels={c} 不一致，请调整输入层或数据。")
    elif spec.mode == "csv":
        x, y = load_csv_nchw_labels(
            spec.primary,
            channels=c,
            height=h,
            width=w,
            skip_header_row=spec.csv_skip_header,
        )
    elif spec.mode == "npy_pair":
        x, y = load_npy_training_pair(spec.primary, spec.secondary)
    else:
        raise ValueError(f"未知训练数据模式: {spec.mode}")

    if int(y.max().item()) >= k or int(y.min().item()) < 0:
        raise ValueError(f"标签需在 [0, {k - 1}] 内。")
    return x, y


def make_train_dataloader_from_graph(
    doc: GraphDocument,
    *,
    shuffle: bool = True,
) -> Any:
    """按 Input.batch 构造 ``DataLoader``（与导出训练脚本一致：批训练、shuffle）。"""
    from dl_vis.logic.graph_dataset import describe_graph_training_hint, parse_graph_linked_training

    spec = parse_graph_linked_training(doc)
    if spec is None:
        raise ValueError(describe_graph_training_hint(doc))
    err = validate_output_for_train(doc)
    if err:
        raise ValueError(err)
    k, kerr = effective_num_classes(doc)
    if kerr or k is None:
        raise ValueError(kerr or "无法确定类别数 K。")
    batch, c, h, w = _input_nchw(doc)
    chk = check_graph_runnable(doc)
    if not chk.ok:
        raise ValueError(chk.error or chk.shape.message)

    bs = max(1, int(batch))

    if spec.mode == "image_folder":
        try:
            from torchvision import datasets
        except ImportError as e:
            raise ImportError("加载图像文件夹需要 torchvision，请执行：pip install torchvision") from e
        tfm = build_imagefolder_tv_transforms(c, h, w)
        ds = datasets.ImageFolder(spec.primary, transform=tfm)
        if len(ds) == 0:
            raise ValueError("ImageFolder 目录下无可用样本。")
        if len(ds.classes) != int(k):
            raise ValueError(
                f"子文件夹类别数 {len(ds.classes)} 与图中 NUM_CLASSES={k} 不一致。"
            )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)

    if spec.mode == "csv":
        x, y = load_csv_nchw_labels(
            spec.primary,
            channels=c,
            height=h,
            width=w,
            skip_header_row=spec.csv_skip_header,
        )
    elif spec.mode == "npy_pair":
        x, y = load_npy_training_pair(spec.primary, spec.secondary)
    else:
        raise ValueError(f"未知训练数据模式: {spec.mode}")

    if int(y.max().item()) >= k or int(y.min().item()) < 0:
        raise ValueError(f"标签需在 [0, {k - 1}] 内。")
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle)


def train_with_dataloader(
    doc: GraphDocument,
    loader: Any,
    *,
    epochs: int,
    lr: float = 1e-3,
    skip_trailing_softmax: bool = True,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]:
    """按 batch 做 CE 训练（与导出脚本的循环结构一致）。"""
    import torch
    import torch.nn as nn

    err = validate_output_for_train(doc)
    if err:
        raise ValueError(err)
    k, err2 = effective_num_classes(doc)
    if err2 or k is None:
        raise ValueError(err2 or "无法确定类别数 K。")
    chk = check_graph_runnable(doc)
    if not chk.ok:
        raise ValueError(chk.error or chk.shape.message)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sequential(doc, skip_trailing_softmax=skip_trailing_softmax)
    model.train()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses: list[float] = []
    for ep in range(max(1, int(epochs))):
        tot = 0.0
        nb = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            if logits.dim() == 4:
                logits = logits.view(logits.size(0), -1)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach().item())
            nb += 1
        avg = tot / max(1, nb)
        losses.append(avg)
        if on_epoch is not None:
            on_epoch(ep + 1, avg)
    return losses


def train_from_graph_dataset(
    doc: GraphDocument,
    *,
    epochs: int,
    lr: float = 1e-3,
    skip_trailing_softmax: bool = True,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]:
    """使用 Dataset→Input 解析到的数据做 CE 训练（DataLoader + 按 batch，与导出一致）。"""
    loader = make_train_dataloader_from_graph(doc)
    return train_with_dataloader(
        doc,
        loader,
        epochs=epochs,
        lr=lr,
        skip_trailing_softmax=skip_trailing_softmax,
        on_epoch=on_epoch,
    )
