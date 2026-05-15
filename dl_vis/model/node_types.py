"""节点类型枚举、默认参数与可编辑字段定义。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Union

# 可编辑字段：``(键, 类型)`` 或 ``(键, "choice", (选项…))``
EditableFieldSpec = Union[tuple[str, str], tuple[str, str, tuple[str, ...]]]


class NodeType(str, Enum):
    INPUT = "Input"
    OUTPUT = "Output"
    DATASET = "Dataset"
    CONV3X3 = "Conv3x3"
    CONV1X1 = "Conv1x1"
    MAX_POOL = "MaxPool"
    AVG_POOL = "AvgPool"
    FC = "FC"
    RELU = "ReLU"
    SIGMOID = "Sigmoid"
    SOFTMAX = "Softmax"
    BN = "BN"
    ADD = "Add"
    CONCAT = "Concat"
    MULTIPLY = "Multiply"
    HIST_EQUALIZE = "HistEqualize"
    MEL_SPECTROGRAM = "MelSpectrogram"
    VIDEO_FRAME_PACK = "VideoFramePack"
    # 占位：仅 UI + 默认参数
    RESIDUAL = "Residual"
    PRUNE = "Prune"
    ATTENTION = "Attention"


# 每种类型默认超参（可被文档覆盖）
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    NodeType.INPUT.value: {
        "batch": 1,
        "channels": 3,
        "height": 224,
        "width": 224,
    },
    NodeType.OUTPUT.value: {
        "name": "logits",
        "task": "classify",
        "num_classes": 0,
        "loss": "cross_entropy",
        "train_epochs": 20,
        "train_lr": 1e-3,
    },
    NodeType.DATASET.value: {
        "path": "",
        "path_kind": "file",
        "role": "unspecified",
        "csv_skip_header": False,
    },
    NodeType.HIST_EQUALIZE.value: {},
    NodeType.MEL_SPECTROGRAM.value: {
        "n_mels": 64,
        "mel_width": 224,
        "n_fft": 1024,
        "hop_length": 256,
        "audio_sample_rate": 16000,
    },
    NodeType.VIDEO_FRAME_PACK.value: {
        "max_frames": 8,
        "out_height": 224,
        "out_width": 224,
        "bad_frame_var_threshold": 1e-5,
    },
    NodeType.CONV3X3.value: {
        "in_channels": 3,
        "out_channels": 64,
        "stride": 1,
        "padding": 1,
        "bias": True,
    },
    NodeType.CONV1X1.value: {
        "in_channels": 64,
        "out_channels": 64,
        "stride": 1,
        "padding": 0,
        "bias": True,
    },
    NodeType.MAX_POOL.value: {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
    },
    NodeType.AVG_POOL.value: {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
    },
    NodeType.FC.value: {
        "in_features": 512,
        "out_features": 10,
        "bias": True,
    },
    NodeType.RELU.value: {
        "inplace": False,
    },
    NodeType.SIGMOID.value: {},
    NodeType.SOFTMAX.value: {
        "dim": -1,
    },
    NodeType.BN.value: {
        "num_features": 64,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
    },
    NodeType.ADD.value: {},
    NodeType.CONCAT.value: {
        "concat_dim": 1,
    },
    NodeType.MULTIPLY.value: {},
    NodeType.RESIDUAL.value: {
        "placeholder": True,
        "note": "第二阶段展开为 Module",
    },
    NodeType.PRUNE.value: {
        "placeholder": True,
        "sparsity": 0.5,
    },
    NodeType.ATTENTION.value: {
        "placeholder": True,
        "embed_dim": 256,
        "num_heads": 8,
    },
}

# UI 动态表单字段：(参数名, 控件类型)
# 类型: int | float | bool | str
EDITABLE_FIELDS: dict[str, list[EditableFieldSpec]] = {
    NodeType.INPUT.value: [
        ("batch", "int"),
        ("channels", "int"),
        ("height", "int"),
        ("width", "int"),
    ],
    NodeType.OUTPUT.value: [
        ("name", "str"),
        ("task", "choice", ("classify",)),
        ("num_classes", "int"),
        ("loss", "choice", ("cross_entropy",)),
        ("train_epochs", "int"),
        ("train_lr", "float"),
    ],
    NodeType.DATASET.value: [
        ("path", "str"),
        (
            "role",
            "choice",
            ("unspecified", "to_input", "from_output"),
        ),
        ("csv_skip_header", "bool"),
    ],
    NodeType.HIST_EQUALIZE.value: [],
    NodeType.MEL_SPECTROGRAM.value: [
        ("n_mels", "int"),
        ("mel_width", "int"),
        ("n_fft", "int"),
        ("hop_length", "int"),
        ("audio_sample_rate", "int"),
    ],
    NodeType.VIDEO_FRAME_PACK.value: [
        ("max_frames", "int"),
        ("out_height", "int"),
        ("out_width", "int"),
        ("bad_frame_var_threshold", "float"),
    ],
    NodeType.CONV3X3.value: [
        ("in_channels", "int"),
        ("out_channels", "int"),
        ("stride", "int"),
        ("padding", "int"),
        ("bias", "bool"),
    ],
    NodeType.CONV1X1.value: [
        ("in_channels", "int"),
        ("out_channels", "int"),
        ("stride", "int"),
        ("padding", "int"),
        ("bias", "bool"),
    ],
    NodeType.MAX_POOL.value: [
        ("kernel_size", "int"),
        ("stride", "int"),
        ("padding", "int"),
    ],
    NodeType.AVG_POOL.value: [
        ("kernel_size", "int"),
        ("stride", "int"),
        ("padding", "int"),
    ],
    NodeType.FC.value: [
        ("in_features", "int"),
        ("out_features", "int"),
        ("bias", "bool"),
    ],
    NodeType.RELU.value: [
        ("inplace", "bool"),
    ],
    NodeType.SIGMOID.value: [],
    NodeType.SOFTMAX.value: [
        ("dim", "int"),
    ],
    NodeType.BN.value: [
        ("num_features", "int"),
        ("eps", "float"),
        ("momentum", "float"),
        ("affine", "bool"),
    ],
    NodeType.ADD.value: [],
    NodeType.CONCAT.value: [
        ("concat_dim", "int"),
    ],
    NodeType.MULTIPLY.value: [],
    NodeType.RESIDUAL.value: [
        ("placeholder", "bool"),
        ("note", "str"),
    ],
    NodeType.PRUNE.value: [
        ("placeholder", "bool"),
        ("sparsity", "float"),
    ],
    NodeType.ATTENTION.value: [
        ("placeholder", "bool"),
        ("embed_dim", "int"),
        ("num_heads", "int"),
    ],
}


def default_params_for_type(node_type: str) -> dict[str, Any]:
    """返回某类型的参数副本。"""
    base = DEFAULT_PARAMS.get(node_type)
    if base is None:
        return {}
    return dict(base)


PREPROC_PALETTE_ORDER: tuple[str, ...] = (
    NodeType.HIST_EQUALIZE.value,
    NodeType.MEL_SPECTROGRAM.value,
    NodeType.VIDEO_FRAME_PACK.value,
)
PREPROC_PALETTE_TYPES: frozenset[str] = frozenset(PREPROC_PALETTE_ORDER)


def preproc_palette_types() -> list[str]:
    """「数据与预处理」侧栏可拖拽的节点类型（顺序固定）。"""
    return list(PREPROC_PALETTE_ORDER)


def is_preproc_palette_type(node_type: str) -> bool:
    return node_type in PREPROC_PALETTE_TYPES


def palette_types() -> list[str]:
    """主「算子」调色板：排除 Dataset 与预处理专用节点。"""
    skip = {NodeType.DATASET.value, *PREPROC_PALETTE_TYPES}
    return [m.value for m in NodeType if m.value not in skip]


def is_known_type(node_type: str) -> bool:
    return node_type in DEFAULT_PARAMS


# 调色板显示名（中文） + 悬停说明（结构/参数/连接方式）；类型键仍为英文以便序列化
NODE_PALETTE_ZH: dict[str, tuple[str, str]] = {
    NodeType.INPUT.value: (
        "输入层",
        "【类型标识】Input\n"
        "【作用】声明整张图的入口张量形状，占位表示一批图像或特征。\n"
        "【张量语义】参数约定为 NCHW：批次 batch、通道 channels、高 height、宽 width。\n"
        "【参数】batch / channels / height / width。\n"
        "【连接】仅有输出端口（右侧）；下游接单入边的第一个算子。\n"
        "【说明】形状推导占位逻辑要求全图恰好一个输入节点。",
    ),
    NodeType.DATASET.value: (
        "训练数据集",
        "【类型标识】Dataset\n"
        "【作用】在图中标记训练/评测数据来源（文件或文件夹路径）。\n"
        "【参数】path：绝对或相对路径；role：与模型关联的说明（unspecified/to_input/from_output）。\n"
        "【连接】建议：数据集出口 → Input 入口（数据送入网络）；或 Output 出口 → 数据集入口（标签、落盘或与输出绑定）。\n"
        "【说明】不参与 NCHW 张量推导与 nn.Sequential 导出；导出时自动忽略。",
    ),
    NodeType.OUTPUT.value: (
        "输出头",
        "【类型标识】Output\n"
        "【作用】标记网络的逻辑输出（如 logits / 嵌入），便于导出与可视化。\n"
        "【参数】name：输出名称；task：任务占位（当前仅 classify）；num_classes：类别数（0 表示与链上末层 FC 的 out_features 一致）；loss：损失占位（当前仅 cross_entropy）。\n"
        "【连接】左侧输入端口接上游算子；右侧无输出。\n"
        "【说明】不参与卷积维度推导时可视为透传。",
    ),
    NodeType.CONV3X3.value: (
        "卷积 3×3",
        "【类型标识】Conv3x3\n"
        "【作用】二维卷积，核固定理解为 3×3。\n"
        "【参数】in_channels / out_channels / stride / padding / bias。\n"
        "【维度】占位推导下输出空间尺寸约为 floor((H+2p−3)/s)+1（宽同理）。\n"
        "【连接】左入右出，单输入单输出；与 PyTorch nn.Conv2d(kernel_size=3) 对应。",
    ),
    NodeType.CONV1X1.value: (
        "卷积 1×1",
        "【类型标识】Conv1x1\n"
        "【作用】逐点卷积，常用于升降通道或瓶颈。\n"
        "【参数】同 Conv，核理解为 1×1。\n"
        "【维度】空间尺寸在 stride=1、padding=0 时常与输入相同（占位公式按 1×1 计算）。\n"
        "【连接】左入右出；对应 nn.Conv2d(kernel_size=1)。",
    ),
    NodeType.MAX_POOL.value: (
        "最大池化",
        "【类型标识】MaxPool\n"
        "【作用】滑动窗口内取最大值，降低空间分辨率。\n"
        "【参数】kernel_size / stride / padding。\n"
        "【维度】占位推导：floor((H+2p−k)/s)+1。\n"
        "【连接】左入右出；对应 nn.MaxPool2d。",
    ),
    NodeType.AVG_POOL.value: (
        "平均池化",
        "【类型标识】AvgPool\n"
        "【作用】滑动窗口内取平均值。\n"
        "【参数】kernel_size / stride / padding。\n"
        "【维度】与 MaxPool 占位公式相同。\n"
        "【连接】左入右出；对应 nn.AvgPool2d。",
    ),
    NodeType.FC.value: (
        "全连接",
        "【类型标识】FC（Fully Connected）\n"
        "【作用】将特征向量映射到另一维度，等价线性层。\n"
        "【参数】in_features / out_features / bias。\n"
        "【维度】占位实现将上游 NCHW 压缩为一维语义后输出 (N, out_features, 1, 1) 形式占位。\n"
        "【连接】左入右出；对应 nn.Linear（导出时需展平）。",
    ),
    NodeType.RELU.value: (
        "ReLU",
        "【类型标识】ReLU\n"
        "【作用】逐元素 max(0, x)。\n"
        "【参数】inplace：是否就地运算（仅语义占位）。\n"
        "【维度】形状与输入一致。\n"
        "【连接】左入右出；对应 nn.ReLU。",
    ),
    NodeType.SIGMOID.value: (
        "Sigmoid",
        "【类型标识】Sigmoid\n"
        "【作用】逐元素 σ(x)=1/(1+e^(−x))。\n"
        "【参数】无额外可编辑字段（可在右侧参数面板保持默认）。\n"
        "【维度】形状不变。\n"
        "【连接】左入右出；对应 nn.Sigmoid。",
    ),
    NodeType.SOFTMAX.value: (
        "Softmax",
        "【类型标识】Softmax\n"
        "【作用】在指定维度上做归一化指数。\n"
        "【参数】dim：轴索引（可为负数）。\n"
        "【维度】总元素个数不变，形状不变。\n"
        "【连接】左入右出；对应 nn.Softmax。",
    ),
    NodeType.BN.value: (
        "批归一化",
        "【类型标识】BN（BatchNorm2d 占位）\n"
        "【作用】按通道标准化并仿射变换，稳定训练。\n"
        "【参数】num_features / eps / momentum / affine。\n"
        "【维度】通道维须与 num_features 一致；空间维占位推导中保持不变。\n"
        "【连接】左入右出；对应 nn.BatchNorm2d。",
    ),
    NodeType.ADD.value: (
        "相加（逐元素）",
        "【类型标识】Add\n"
        "【作用】多路同形状 NCHW 逐元素相加（torch.add）。\n"
        "【维度】所有输入的 N、C、H、W 须一致。\n"
        "【连接】至少两路入边指向本节点（边顺序决定与 Concat 以外的约定）；一路出边。",
    ),
    NodeType.CONCAT.value: (
        "拼接 Concat",
        "【类型标识】Concat\n"
        "【作用】沿指定维拼接张量（torch.cat）。\n"
        "【参数】concat_dim：0=N，1=C，2=H，3=W（可为负数，按 4 维语义归一）。\n"
        "【维度】除拼接维外其余维必须一致；输出在拼接维上为各输入之和。",
    ),
    NodeType.MULTIPLY.value: (
        "相乘（逐元素）",
        "【类型标识】Multiply\n"
        "【作用】多路同形状逐元素相乘。\n"
        "【维度】所有输入 N、C、H、W 须一致。",
    ),
    NodeType.HIST_EQUALIZE.value: (
        "直方图均衡",
        "【类型标识】HistEqualize\n"
        "【作用】声明对图像特征做直方图均衡（与 logic.dataproc.image_prep 一致）。\n"
        "【维度】NCHW 不变。\n"
        "【连接】单入单出。\n"
        "【导出】不参与 nn.Sequential；运行时在数据管线中调用等价变换。",
    ),
    NodeType.MEL_SPECTROGRAM.value: (
        "梅尔频谱",
        "【类型标识】MelSpectrogram\n"
        "【作用】波形→对数梅尔谱，输出视作单通道伪图像 (N,1,n_mels,mel_width)。\n"
        "【参数】n_mels / mel_width / n_fft / hop_length / audio_sample_rate。\n"
        "【连接】单入单出；首层 Conv 的 in_channels 应为 1。\n"
        "【导出】不参与 nn.Sequential；见 logic.dataproc.mel_spectrogram。",
    ),
    NodeType.VIDEO_FRAME_PACK.value: (
        "视频抽帧堆叠",
        "【类型标识】VideoFramePack\n"
        "【作用】多帧 RGB 预处理后沿通道拼接，(N, 3×max_frames, out_height, out_width)。\n"
        "【参数】max_frames / out_height / out_width / bad_frame_var_threshold。\n"
        "【连接】单入单出；首层 Conv in_channels = 3×max_frames。\n"
        "【导出】不参与 nn.Sequential；见 logic.dataproc.video_frames。",
    ),
    NodeType.RESIDUAL.value: (
        "残差（占位）",
        "【类型标识】Residual\n"
        "【作用】表示捷径相加结构；当前仅为图上一类节点，无训练逻辑。\n"
        "【参数】placeholder / note（文档说明）。\n"
        "【连接】设计上支持多入一出（后续版本展开）；现阶段可与多条边拓扑预留。\n"
        "【说明】导出 PyTorch 时需在第二阶段展开为自定义 Module。",
    ),
    NodeType.PRUNE.value: (
        "剪枝（占位）",
        "【类型标识】Prune\n"
        "【作用】表示剪枝策略占位；不参与前向数值模拟。\n"
        "【参数】placeholder / sparsity（稀疏比例语义占位）。\n"
        "【连接】左入右出。\n"
        "【说明】第二阶段再接入实际剪枝与训练流程。",
    ),
    NodeType.ATTENTION.value: (
        "注意力（占位）",
        "【类型标识】Attention\n"
        "【作用】表示多头注意力等结构占位。\n"
        "【参数】placeholder / embed_dim / num_heads。\n"
        "【连接】分支与残差将在扩展拓扑中表达。\n"
        "【说明】链式形状推导未实现；导出需第二阶段生成 nn.MultiheadAttention 等。",
    ),
}

# 参数面板字段中文标签（键与 EDITABLE_FIELDS 一致）
PARAM_LABEL_ZH: dict[str, str] = {
    "batch": "批次大小 N",
    "channels": "通道数 C",
    "height": "高度 H",
    "width": "宽度 W",
    "name": "输出名称",
    "task": "任务类型（占位）",
    "num_classes": "类别数 K（0=继承末层 FC）",
    "loss": "损失（占位）",
    "train_epochs": "导出/默认训练轮数",
    "train_lr": "导出/默认学习率",
    "in_channels": "输入通道数",
    "out_channels": "输出通道数",
    "stride": "步幅",
    "padding": "填充",
    "bias": "使用偏置",
    "kernel_size": "池化核大小",
    "in_features": "输入特征维",
    "out_features": "输出特征维",
    "inplace": "就地运算",
    "dim": "Softmax 维度",
    "num_features": "特征通道数",
    "eps": "数值稳定项 ε",
    "momentum": "动量",
    "affine": "可学习仿射",
    "placeholder": "占位标记",
    "note": "备注说明",
    "sparsity": "稀疏比例（占位）",
    "embed_dim": "嵌入维度",
    "num_heads": "注意力头数",
    "concat_dim": "拼接维（0=N,1=C,2=H,3=W）",
    "path": "数据路径（文件或文件夹）",
    "path_kind": "路径类型（file/folder，自动）",
    "role": "与模型的关联（说明）",
    "csv_skip_header": "CSV 首行作表头（跳过）",
    "mel_width": "梅尔时间维宽度 W",
    "n_mels": "梅尔频带数（输出高度 H）",
    "n_fft": "STFT n_fft",
    "hop_length": "STFT hop",
    "audio_sample_rate": "音频采样率",
    "max_frames": "堆叠帧数（输出通道=3×帧数）",
    "out_height": "输出高度 H",
    "out_width": "输出宽度 W",
    "bad_frame_var_threshold": "坏帧方差阈值",
}


def palette_label_zh(node_type: str) -> str:
    return NODE_PALETTE_ZH.get(node_type, (node_type, ""))[0]


def palette_doc_zh(node_type: str) -> str:
    return NODE_PALETTE_ZH.get(node_type, ("", ""))[1]


def param_label_zh(param_key: str) -> str:
    return PARAM_LABEL_ZH.get(param_key, param_key)
