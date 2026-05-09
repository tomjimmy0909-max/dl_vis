# dl_vis 设计说明（MVP）

## 1. 架构分层

| 层级 | 职责 | 第一阶段 |
|------|------|----------|
| 界面层 | `MainWindow`、多 Tab 画布、`QDockWidget` 参数面板 | 已实现 |
| 节点视图 | `NodeItem`、`EdgeItem`、`CanvasWidget` | 已实现 |
| 逻辑层 | `GraphDocument`、形状推导占位、导出占位 | 部分 |
| 框架层 | PyTorch 代码生成 / 执行 | 第二阶段 |

```mermaid
flowchart LR
  UI[PyQt6 UI] --> Scene[GraphicsScene]
  Scene --> Doc[GraphDocument]
  Doc --> Shape[shape_inference]
  Doc --> Export[export_torch]
```

## 2. 图模型

- **约束**：有向无环图（DAG）；添加边后若产生环则拒绝。
- **节点**：全局唯一 `id`（UUID 字符串）、`type`、`x`/`y` 场景坐标、`params` 字典。
- **边**：唯一 `id`、`src_id`、`dst_id`，可选 `src_port` / `dst_port`（默认 `out` / `in`），便于残差与多输入。
- **校验**：禁止自环；重复 `(src_id, dst_id)` 不允许。

## 3. JSON 序列化 Schema（草案）

文件根对象：

```json
{
  "schema_version": "1.0",
  "nodes": [
    {
      "id": "uuid",
      "type": "Conv3x3",
      "x": 120.0,
      "y": 80.0,
      "params": { "in_channels": 3, "out_channels": 64 }
    }
  ],
  "edges": [
    {
      "id": "uuid",
      "src_id": "...",
      "dst_id": "...",
      "src_port": "out",
      "dst_port": "in"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | string | 当前 `1.0` |
| `nodes[].id` | string | 必填 |
| `nodes[].type` | string | 见下表 |
| `nodes[].x`, `y` | number | 场景坐标 |
| `nodes[].params` | object | 类型相关 |
| `edges[].id` | string | 必填 |
| `edges[].src_id`, `dst_id` | string | 必填 |

## 4. 节点类型与默认参数（第一阶段）

| type | 说明 | 备注 |
|------|------|------|
| Input | NCHW 占位 | batch/channels/height/width |
| Output | 输出头占位 | name |
| Conv3x3 / Conv1x1 | 卷积 | stride/padding/bias |
| MaxPool / AvgPool | 池化 | kernel/stride/padding |
| FC | 全连接 | in/out features |
| ReLU / Sigmoid / Softmax | 激活 | 按类型 |
| BN | BatchNorm | num_features 等 |
| Residual / Prune / Attention | 占位 | 无训练逻辑，参数仅文档/UI |

详细默认键值见 `dl_vis/model/node_types.py` 中 `DEFAULT_PARAMS` 与 `EDITABLE_FIELDS`。

## 5. 坐标与交互约定

- 节点矩形宽约 140×70（逻辑单位）；输出锚点为右侧中点，输入为左侧中点。
- 连线为场景坐标下的折线/贝塞尔路径，随节点移动更新。

## 6. 与 PyTorch 映射（预留）

- **Sequential**：线性链可映射为 `nn.Sequential`。
- **分支**：多条入边在第二阶段通过自定义 `Module` 或 `forward` 拼接语义生成代码。
- **Residual / Attention**：图中为独立节点类型；代码生成阶段展开（第一阶段不实现）。

## 7. 扩展（后续）

- **自定义算子**：注册表 `type → 参数 schema + 可选 forward 钩子`。
- **插件**：动态加载 Python 模块注册节点类型。

## 8. 阶段路线图

| 阶段 | 内容 |
|------|------|
| **当前 MVP** | 多 Tab、拖拽节点、连线、参数 Dock、JSON 存盘、形状推导占位、导出菜单占位 |
| **二（部分已落地）** | **已实现**：`QUndoStack` 快照撤销/重做；多选节点左/顶对齐；线性链 `nn.Sequential` 源码复制/保存；「可视化」Tab 中选中节点的参数条形图；形状推导一档（线性链 NCHW + FC `in_features` 占位校验与 warnings）及二档错误提示文案。**仍扩展中**：任意 DAG 形状推导、分支/汇合导出、完整 Matplotlib 热力图等。 |
| **三** | 训练子图、梯度钩子、插件加载 |

### 8.1 已实现能力摘要（形状推导）

- **一档**：图中恰好一个 `Input`，且无分叉、无汇合（单路径）；沿链传播 NCHW；`FC` 若 `in_features ≠ C×H×W` 则记入 `warnings`。
- **二档**：分叉或汇合时返回明确中文说明（尚未做逐路径传播）。
- **三档**：规划中。

## 9. 执行子图与可视化（第三阶段语义草案）

以下为第三阶段实现前的约定草案，便于接口对齐。

### 9.1 GraphExecutor（草案）

- **子图选定**：由用户指定节点集合 `S ⊆ nodes`，要求从「损失/输出」反向可达或从前向入口正向闭合（具体策略在实现时固定为一种）。
- **前向**：按拓扑序对 `S` 内节点执行注册的 `forward` 钩子或内置算子映射，张量在节点之间沿边传递。
- **反向**：在 `S` 上对标量损失调用 `backward()`；若节点不在 `S` 内则不参与求导。
- **与画布关系**：`GraphDocument` 仍为权威拓扑；Executor 只读文档 + 运行时缓存。

### 9.2 hook（草案）

- **注册**：节点类型或实例可登记 `forward_hook(ctx) -> Tensor | None`，由执行器在前后向时调用。
- **可视化订阅**：UI/Matplotlib 层订阅 hook 暴露的中间张量引用（弱引用或句柄），用于热力图、梯度范数等；**不在第三阶段前强制实现**。

### 9.3 插件（草案）

- 动态 `importlib` 加载模块，调用约定入口（如 `register_nodes(registry)`）向全局注册表追加 `type`、默认参数 schema 与可选 `forward`。
