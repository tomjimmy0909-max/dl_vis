"""graph_dataset 与导出训练脚本（轻量）测试。"""

from __future__ import annotations

import unittest

from dl_vis.logic.export_torch import export_full_training_script
from dl_vis.logic.graph_dataset import datasets_feeding_input, parse_graph_linked_training
from dl_vis.model.graph_document import GraphDocument
from dl_vis.model.node_types import NodeType


class TestGraphDataset(unittest.TestCase):
    def test_parse_folder_to_input(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0)
        ds = d.add_node(
            NodeType.DATASET.value,
            0,
            0,
            params={"path": "D:/data_root", "path_kind": "folder", "role": "to_input"},
        )
        d.add_edge(ds.id, inp.id)
        spec = parse_graph_linked_training(d)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.mode, "image_folder")
        self.assertIn("data_root", spec.primary)

    def test_parse_two_npy(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0)
        a = d.add_node(NodeType.DATASET.value, 0, 0, params={"path": "/tmp/b_y.npy"})
        b = d.add_node(NodeType.DATASET.value, 0, 0, params={"path": "/tmp/a_x.npy"})
        d.add_edge(a.id, inp.id)
        d.add_edge(b.id, inp.id)
        spec = parse_graph_linked_training(d)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.mode, "npy_pair")
        self.assertTrue(spec.primary.endswith("a_x.npy"))
        self.assertTrue(spec.secondary.endswith("b_y.npy"))

    def test_datasets_feeding_input_order(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0)
        ds = d.add_node(NodeType.DATASET.value, 0, 0, params={"path": "x"})
        d.add_edge(ds.id, inp.id)
        self.assertEqual(len(datasets_feeding_input(d)), 1)


class TestExportTrainingScript(unittest.TestCase):
    def test_export_appends_training_block(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0, params={"batch": 1, "channels": 1, "height": 4, "width": 4})
        fc = d.add_node(
            NodeType.FC.value, 0, 0, params={"in_features": 16, "out_features": 2, "bias": True}
        )
        out = d.add_node(NodeType.OUTPUT.value, 0, 0, params={"num_classes": 0})
        ds = d.add_node(
            NodeType.DATASET.value,
            0,
            0,
            params={"path": "D:/set", "path_kind": "folder"},
        )
        d.add_edge(ds.id, inp.id)
        d.add_edge(inp.id, fc.id)
        d.add_edge(fc.id, out.id)
        src = export_full_training_script(d)
        self.assertIn("nn.Sequential", src)
        self.assertIn("build_train_dataloader", src)
        self.assertIn("def train(", src)
        self.assertIn("ImageFolder", src)


if __name__ == "__main__":
    unittest.main()
