# CFA-Mamba 公开代码说明

**论文题名：** CFA-Mamba: Frozen CLIP Semantics for Source-Free Domain Adaptation with Mamba.

## 代码说明

本目录是可独立上传 GitHub 的代码包，包含四个数据集的主实验，以及期刊新增的结构对照、
固定 CLIP/source/ensemble 参考、配对统计、预测分析和运行开销测量。

- 算法：`repository/cfa_mamba/model.py`、`losses.py`。
- CLIP 概率缓存：`repository/cfa_mamba/clip_teacher.py`。
- 主实验：`scripts/reproduce_*.sh`。
- 新增对照：`experiments/fixed_budget/`。
- 安装：`INSTALL.md`；数据和权重：`docs/DATA_AND_WEIGHTS.md`。
- 论文实验与参数对应：`docs/REPRODUCIBILITY.md`。

公开版明确采用 OpenCLIP **ViT-B-16 / laion2b_s34b_b88k**，温度 0.01、五个固定模板。
训练中冻结 CLIP、源参考模型和源分类器；学习目标特征提取器、融合模块及适配器。
历史类名中的 Teacher/Student 保留兼容性，不代表论文主张教师—学生蒸馏模型。

建议先按英文 README 的单方向命令运行，再启动完整矩阵。服务器训练需 Linux、NVIDIA GPU、
CUDA 工具链和已编译的 selective-scan 扩展。本次整理只做静态检查与可执行的 CPU 检查，
不把这些检查当成 GPU 复现实验。

主结果取固定训练预算后的最后一次 evaluation；不要取日志最大准确率，也不要将 best checkpoint
视为最终模型。需导出最终模型时，可设置 `TRAIN_ENTRY=train_cg_mamba_save_endpoint.py`。
新增研究的入口自动记录 endpoint、代码哈希、源权重、类别顺序、数据索引和运行环境。

数据集、预训练权重、训练检查点、缓存、原始日志和私人路径未放入代码包。所有路径由环境变量
或 `experiments/fixed_budget/settings.json` 配置。不要把旧工作区加入 PYTHONPATH。

原始论文、代码归档与实验结果未修改。发布整理中的改动见 `docs/RELEASE_CHANGES.md`。
本仓库仅供参考。作者原创部分未授予使用许可，保留所有权利。本仓库未采用开源许可证。
第三方组件仍遵循各自的许可证，本声明不限制这些许可证已授予的权利，也不影响适用法律
及 GitHub 平台条款下的权利。第三方声明及上游许可文本见 `THIRD_PARTY_NOTICES.md`。
上传时将本目录中的文件作为仓库根目录即可，无需一并上传实验工作区。
