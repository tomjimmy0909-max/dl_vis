"""选中节点时右侧「代码检视」：按当前参数生成与导出一致的 PyTorch 片段说明。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dl_vis.model.node_types import NodeType

if TYPE_CHECKING:
    from dl_vis.model.graph_document import GraphNode


def _bool_py(b: Any) -> str:
    return "True" if bool(b) else "False"


def code_preview_for_node(node: GraphNode) -> str:
    """返回可多行只读展示的伪代码 / nn.* 调用串（与 export_torch 语义对齐）。"""
    t = node.type
    p = dict(node.params)
    hdr = f"# 节点类型: {t}\n"

    if t == NodeType.INPUT.value:
        return (
            hdr
            + "# 张量形状占位 (N, C, H, W)\n"
            f"x = torch.randn({int(p.get('batch', 1))}, {int(p.get('channels', 3))}, "
            f"{int(p.get('height', 224))}, {int(p.get('width', 224))}, device=device)"
        )

    if t == NodeType.OUTPUT.value:
        return (
            hdr
            + "# 逻辑输出头（不单独对应 nn.Module 层）\n"
            f"# name={p.get('name','')!r}, task={p.get('task','')!r}, "
            f"num_classes={int(p.get('num_classes', 0))}, loss={p.get('loss','')!r}\n"
            f"# 导出训练脚本默认超参: train_epochs={int(p.get('train_epochs', 20))}, "
            f"train_lr={float(p.get('train_lr', 1e-3))}（可用命令行 --epochs/--lr 覆盖）"
        )

    if t == NodeType.DATASET.value:
        skip = bool(p.get("csv_skip_header", False))
        return (
            hdr
            + f"# path_kind={p.get('path_kind', 'file')!r}\n"
            f"path = {p.get('path', '')!r}\n"
            f"# csv 训练时跳过表头: csv_skip_header={_bool_py(skip)}\n"
            f"# role={p.get('role', 'unspecified')!r} "
            "（unspecified=仅说明；to_input=数据侧；from_output=与输出/标签关联）\n"
            "# 建议： Dataset → Input；或 Output → Dataset"
        )

    if t == NodeType.CONV3X3.value:
        ic, oc = int(p["in_channels"]), int(p["out_channels"])
        s, pad = int(p.get("stride", 1)), int(p.get("padding", 0))
        bias = bool(p.get("bias", True))
        return (
            hdr
            + f"nn.Conv2d({ic}, {oc}, kernel_size=3, stride={s}, padding={pad}, bias={_bool_py(bias)})"
        )

    if t == NodeType.CONV1X1.value:
        ic, oc = int(p["in_channels"]), int(p["out_channels"])
        s, pad = int(p.get("stride", 1)), int(p.get("padding", 0))
        bias = bool(p.get("bias", True))
        return (
            hdr
            + f"nn.Conv2d({ic}, {oc}, kernel_size=1, stride={s}, padding={pad}, bias={_bool_py(bias)})"
        )

    if t == NodeType.MAX_POOL.value:
        ks = int(p.get("kernel_size", 2))
        st = int(p.get("stride", ks))
        pad = int(p.get("padding", 0))
        return hdr + f"nn.MaxPool2d(kernel_size={ks}, stride={st}, padding={pad})"

    if t == NodeType.AVG_POOL.value:
        ks = int(p.get("kernel_size", 2))
        st = int(p.get("stride", ks))
        pad = int(p.get("padding", 0))
        return hdr + f"nn.AvgPool2d(kernel_size={ks}, stride={st}, padding={pad})"

    if t == NodeType.FC.value:
        inf, outf = int(p["in_features"]), int(p["out_features"])
        bias = bool(p.get("bias", True))
        return hdr + f"nn.Linear({inf}, {outf}, bias={_bool_py(bias)})"

    if t == NodeType.RELU.value:
        ip = bool(p.get("inplace", False))
        return hdr + f"nn.ReLU(inplace={_bool_py(ip)})"

    if t == NodeType.SIGMOID.value:
        return hdr + "nn.Sigmoid()"

    if t == NodeType.SOFTMAX.value:
        dim = int(p.get("dim", -1))
        return hdr + f"nn.Softmax(dim={dim})"

    if t == NodeType.BN.value:
        nf = int(p["num_features"])
        eps = float(p.get("eps", 1e-5))
        momentum = float(p.get("momentum", 0.1))
        affine = bool(p.get("affine", True))
        return (
            hdr
            + f"nn.BatchNorm2d({nf}, eps={eps}, momentum={momentum}, affine={_bool_py(affine)})"
        )

    if t == NodeType.ADD.value:
        return hdr + "torch.add(x1, x2)  # 或 x1 + x2；两路输入形状须一致"

    if t == NodeType.MULTIPLY.value:
        return hdr + "torch.mul(x1, x2)  # 或 x1 * x2；两路输入形状须一致"

    if t == NodeType.HIST_EQUALIZE.value:
        return (
            hdr
            + "# NCHW 形状不变；运行时用 logic.dataproc.image_prep.chw_uint8_to_tensor_prep(..., use_hist_equalize=True)"
        )

    if t == NodeType.MEL_SPECTROGRAM.value:
        return (
            hdr
            + "# 输出形状 (N,1,n_mels,mel_width)；运行时用 logic.dataproc.mel_spectrogram 按节点参数构造 STFT/梅尔\n"
            f"# n_mels={int(p.get('n_mels', 64))}, mel_width={int(p.get('mel_width', 224))}\n"
            f"# n_fft={int(p.get('n_fft', 1024))}, hop_length={int(p.get('hop_length', 256))}, sr={int(p.get('audio_sample_rate', 16000))}"
        )

    if t == NodeType.VIDEO_FRAME_PACK.value:
        return (
            hdr
            + "# 输出 (N,3×max_frames,out_h,out_w)；运行时用 logic.dataproc.video_frames.video_clip_to_nchw\n"
            f"# max_frames={int(p.get('max_frames', 8))}, out_height={int(p.get('out_height', 224))}, out_width={int(p.get('out_width', 224))}"
        )

    if t == NodeType.CONCAT.value:
        d = int(p.get("concat_dim", 1))
        return hdr + f"torch.cat([tensors...], dim={d})  # NCHW 下 dim=1 为沿通道拼接"

    if t == NodeType.RESIDUAL.value:
        return hdr + "# 残差占位；导出时需展开为 x + branch(x)"

    if t == NodeType.PRUNE.value:
        return hdr + f"# 剪枝占位 sparsity={p.get('sparsity', '')}"

    if t == NodeType.ATTENTION.value:
        return (
            hdr
            + f"# Attention 占位 embed_dim={p.get('embed_dim','')}, "
            f"num_heads={p.get('num_heads','')}"
        )

    return hdr + f"# 暂无代码模板: {t}"
