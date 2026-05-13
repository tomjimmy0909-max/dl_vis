"""形状 DAG 推导与拓扑排序单元测试（unittest）。"""

from __future__ import annotations

import unittest

from dl_vis.logic.export_torch import linear_chain_order
from dl_vis.logic.shape_inference import infer_shapes_dag_nchw
from dl_vis.logic.topo_sort import has_cycle, topological_sort
from dl_vis.model.graph_document import GraphDocument
from dl_vis.model.node_types import NodeType


class TestTopoSort(unittest.TestCase):
    def test_chain_order(self) -> None:
        d = GraphDocument()
        n0 = d.add_node(NodeType.INPUT.value, 0, 0)
        n1 = d.add_node(NodeType.RELU.value, 1, 0)
        n2 = d.add_node(NodeType.OUTPUT.value, 2, 0)
        d.add_edge(n0.id, n1.id)
        d.add_edge(n1.id, n2.id)
        order = topological_sort(d)
        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.index(n0.id) < order.index(n1.id), True)
        self.assertFalse(has_cycle(d))

    def test_cycle(self) -> None:
        d = GraphDocument.from_dict(
            {
                "schema_version": "1.0",
                "nodes": [
                    {
                        "id": "a",
                        "type": "Input",
                        "x": 0,
                        "y": 0,
                        "params": {"batch": 1, "channels": 1, "height": 1, "width": 1},
                    },
                    {"id": "b", "type": "ReLU", "x": 0, "y": 0, "params": {"inplace": False}},
                ],
                "edges": [
                    {"id": "e1", "src_id": "a", "dst_id": "b", "src_port": "out", "dst_port": "in"},
                    {"id": "e2", "src_id": "b", "dst_id": "a", "src_port": "out", "dst_port": "in"},
                ],
            }
        )
        self.assertIsNone(topological_sort(d))
        self.assertTrue(has_cycle(d))


class TestShapeDAG(unittest.TestCase):
    def test_fork_add_ok(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0, params={"batch": 1, "channels": 3, "height": 8, "width": 8})
        b1 = d.add_node(NodeType.RELU.value, 1, 0)
        b2 = d.add_node(NodeType.RELU.value, 2, 0)
        add = d.add_node(NodeType.ADD.value, 3, 0)
        out = d.add_node(NodeType.OUTPUT.value, 4, 0)
        d.add_edge(inp.id, b1.id)
        d.add_edge(inp.id, b2.id)
        d.add_edge(b1.id, add.id)
        d.add_edge(b2.id, add.id)
        d.add_edge(add.id, out.id)
        r = infer_shapes_dag_nchw(d)
        self.assertTrue(r.ok)
        assert r.shapes_by_node is not None
        self.assertEqual(r.shapes_by_node[add.id], (1, 3, 8, 8))

    def test_add_mismatch(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0, params={"batch": 1, "channels": 3, "height": 8, "width": 8})
        c1 = d.add_node(NodeType.CONV3X3.value, 1, 0, params={"in_channels": 3, "out_channels": 16, "stride": 1, "padding": 1, "bias": True})
        c2 = d.add_node(NodeType.CONV3X3.value, 2, 0, params={"in_channels": 3, "out_channels": 32, "stride": 2, "padding": 0, "bias": True})
        add = d.add_node(NodeType.ADD.value, 3, 0)
        out = d.add_node(NodeType.OUTPUT.value, 4, 0)
        d.add_edge(inp.id, c1.id)
        d.add_edge(inp.id, c2.id)
        d.add_edge(c1.id, add.id)
        d.add_edge(c2.id, add.id)
        d.add_edge(add.id, out.id)
        r = infer_shapes_dag_nchw(d)
        self.assertFalse(r.ok)
        self.assertIn("Add", r.message)

    def test_concat_channels(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0, params={"batch": 1, "channels": 3, "height": 4, "width": 4})
        c1 = d.add_node(NodeType.CONV1X1.value, 1, 0, params={"in_channels": 3, "out_channels": 8, "stride": 1, "padding": 0, "bias": True})
        c2 = d.add_node(NodeType.CONV1X1.value, 2, 0, params={"in_channels": 3, "out_channels": 8, "stride": 1, "padding": 0, "bias": True})
        cat = d.add_node(NodeType.CONCAT.value, 3, 0, params={"concat_dim": 1})
        out = d.add_node(NodeType.OUTPUT.value, 4, 0)
        d.add_edge(inp.id, c1.id)
        d.add_edge(inp.id, c2.id)
        d.add_edge(c1.id, cat.id)
        d.add_edge(c2.id, cat.id)
        d.add_edge(cat.id, out.id)
        r = infer_shapes_dag_nchw(d)
        self.assertTrue(r.ok)
        assert r.shapes_by_node is not None
        self.assertEqual(r.shapes_by_node[cat.id][1], 16)

    def test_dataset_ignored_in_shape_chain(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0, params={"batch": 1, "channels": 3, "height": 4, "width": 4})
        relu = d.add_node(NodeType.RELU.value, 1, 0)
        out = d.add_node(NodeType.OUTPUT.value, 2, 0)
        ds = d.add_node(NodeType.DATASET.value, 3, 0, params={"path": "/tmp/x", "path_kind": "folder", "role": "to_input"})
        d.add_edge(inp.id, relu.id)
        d.add_edge(relu.id, out.id)
        d.add_edge(ds.id, inp.id)
        r = infer_shapes_dag_nchw(d)
        self.assertTrue(r.ok)

    def test_linear_chain_skips_dataset(self) -> None:
        d = GraphDocument()
        inp = d.add_node(NodeType.INPUT.value, 0, 0)
        relu = d.add_node(NodeType.RELU.value, 1, 0)
        out = d.add_node(NodeType.OUTPUT.value, 2, 0)
        ds = d.add_node(NodeType.DATASET.value, 3, 0, params={"path": "D:/data", "path_kind": "folder"})
        d.add_edge(inp.id, relu.id)
        d.add_edge(relu.id, out.id)
        d.add_edge(out.id, ds.id)
        order = linear_chain_order(d)
        self.assertEqual(len(order), 3)
        self.assertNotIn(ds.id, order)


if __name__ == "__main__":
    unittest.main()
