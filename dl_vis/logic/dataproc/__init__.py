"""多模态数据：统一 NCHW 张量、时间对齐、预处理与数据集划分（逻辑子模块，不替代 graph_dataset）。"""

from dl_vis.logic.dataproc.align import align_nearest_pairs, resample_index_map
from dl_vis.logic.dataproc.config import DataProcConfig
from dl_vis.logic.dataproc.pipeline import (
    bgr_frame_to_nchw,
    guess_modality,
    process_path_to_nchw,
    tensor_nchw_summary,
)
from dl_vis.logic.dataproc.split import split_indices_train_val_test
from dl_vis.logic.dataproc.webcam_stream import WebcamFrameQueue

__all__ = [
    "DataProcConfig",
    "WebcamFrameQueue",
    "align_nearest_pairs",
    "resample_index_map",
    "bgr_frame_to_nchw",
    "guess_modality",
    "split_indices_train_val_test",
    "process_path_to_nchw",
    "tensor_nchw_summary",
]
