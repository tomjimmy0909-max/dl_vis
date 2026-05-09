# dl_vis — 深度学习可视化建模（MVP）

基于 PyQt6 的桌面工具：多标签画布、拖拽节点、连线编辑、参数面板与 JSON 图序列化。

## 环境（Windows）

在项目根目录 `dl_vis/` 下使用虚拟环境（与工作区中的 CUDA/VS 工程互不干扰）。

```powershell
cd dl_vis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 启动

在已激活的 venv 中，从 **`dl_vis` 目录**（含 `requirements.txt` 的目录）执行：

```powershell
python -m dl_vis
```

也可将 `dl_vis` 包所在路径加入 `PYTHONPATH` 后运行同一模块。

## 操作摘要

- **_palette_**：从左侧类型列表拖到画布创建节点。
- **连线**：从节点右侧输出端口按下拖到另一节点左侧输入端口释放。
- **删除**：选中节点或边后按 `Delete`。
- **保存/打开**：菜单「文件」→ 保存或打开 JSON 图文件。
- **导出 PyTorch**：菜单项预留，当前为未实现提示。

## 目录说明

- `dl_vis/dl_vis/` — Python 包（`app`、`ui`、`model`、`logic`）。
- `dl_vis/docs/DESIGN.md` — 架构与 JSON Schema 说明。
