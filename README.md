# Multi-Technology-Fusion-AI-Based-on-Intelligent-Agent-Architecture
Multi-Technology Fusion AI Based on Intelligent Agent Architecture
# Bayer Process Analytics Platform

## 1. Overview
这个项目面向氧化铝工序，通过 FastAPI + Jinja2 构建一个可视化、可训练、可问答的分析平台。用户可直接上传 CSV/Excel，完成数据缓存、字段提取、模型训练、预测、知识图谱/根因分析与 TableGPT 智能问答的全链路流程，适合在本地或自有服务器上部署做实验室级建模与推理探索。

## 2. Architecture & Data Flow
- **前端**：`app/templates/model_lab.html`（模型实验室）、`parameter_analysis.html`（指标分析）、`kg_root_cause_lab.html`（知识图谱/根因）等页面提供上传/训练/问答/可视化交互，并通过缓存键协调后端。
- **后端 API（`app/main.py`）**： `/model-lab/upload`、`/analysis`、`/model/train`、`/model/predict`、`/table-chat`、`/kg-*` 等接口，利用 `_store_dataframe/_restore_dataframe` 实现数据共享。 `/model-lab/upload` 单独负责文件解析、数据清理与字段元信息返回，避免依赖指标分析入口。
- **平衡计算入口**：首页 `index.html` 在上传区域旁提供“平衡计算”按钮，前端通过 `balance-modal` 收集矿石/石灰/氧化铝等成分及工艺参数，调用后端 `/calculate-equilibrium` 接口（实现于 `services/equilibrium.py`）完成物料/热力平衡预测，同时在页面上展示核心指标和调整建议。
- **数据准备**：`services/preprocessing.py` 中的 `prepare_dataset` 完成 timestamp 检测、数值化、lag 特征选择与 metadata 组装，结果作为模型与分析模块的统一输入。
- **模型训练**：`services/tabular_models.py` 统一 XGBoost/TabPFN/LSTM 的训练接口，返回包含 MSE、MAE、R² 等指标的 `metrics`，并在前端 summary 中展示，提升可追溯性。LSTM 的 R² 由 `services/lstm_model.py` 计算以保持一致。

## 3. Explainability & Analytics
- **SHAP + 因果评分**：`services/kg_root_cause.py` 提取 TreeExplainer 或 XGBoost `pred_contribs` 的 SHAP 贡献，结合 Pearson 相关系数、回归系数与模型重要度在 `_merge_factor_scores` 中生成影响评分，再推送到大屏 `kg_root_cause_lab.html` 的“核心驱动”、“优先措施”、“时序洞察”表格，并可导出建议与调节方向。
- **趋势与相关性**：Knowledge Graph Builder (`services/kg_graph_builder.py`) 构建基于相关性阈值的图谱，支持路径搜索与桥接节点挖掘。前端 `buildSummaryItem`/`renderTrend` 进一步将 `temporal_insights` 及 Granger 分析结果以趋势列表呈现，确保用户能在数值、时序、因果层面理解指标波动。
- **TableGPT 问答**：`services/table_chat.py` 执行字段匹配与缺失字段检测，若提问指标未在数据中即刻报错，从而避免 AI 回答与现实数据脱节。历史对话、数据摘要与结论模板共同构建结构化提示，确保自动化问答输出保持一致性。
- **导出与联动**：`/report/export/excel` 支持将分析结果导出，`/parameter-analysis`、`/agent-payload` 等接口提供额外的数据洞察与自动化代理触发能力。
- **平衡态计算**：`/calculate-equilibrium` 利用 `services/equilibrium` 中的矿石/石灰/氧化铝等成分及工艺参数计算适合作为工艺优化前的初步预测。
- **智能体问答**：`/ai-chat` 提供固定接口，可调用 Bailian/Ollama/TableGPT 等 provider，输入用户问题与报警上下文即可拿到结构化回答，方便在模型实验或生产运行中对接人机协作流程。

## 4. Key Features
- **独立数据上传**：`/model-lab/upload` 直接解析 CSV/XLSX、生成字段元数据与缓存键；前端上传后即可训练，无需先进入指标分析页。
- **统一训练指标**：XGBoost、TabPFN、LSTM 均走同一接口，返回 `train/val` MSE、MAE、R²，并缓存模型 bundle 供预测/问答再利用。
- **可解释知识图谱**：相关性阈值控制、路径搜索、本体节点/边度、Granger/PCA 等信息一并提供给用户做因果与趋势判断。
- **结构化问答保障**：字段存在性检测、提示历史、TableGPT 调用封装，让大模型更可靠地聚焦上传数据、避免幻觉。
- **导出与联动**：`/report/export/excel` 支持将分析结果导出，`/parameter-analysis`、`/agent-payload` 等接口提供额外的数据洞察与自动化代理触发能力。

## 5. Getting Started
1. 创建虚拟环境并安装依赖：
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. 启动应用：
   ```powershell
   python main.py
   ```
3. 打开浏览器访问 `http://localhost:8000/model-lab`，通过“上传数据并启用模型实验”上传 CSV/Excel，测试模型训练与 TableGPT 问答。
4. 若需 TableGPT：设置 `TABLEGPT_API_KEY` 与 `TABLEGPT_BASE_URL` 环境变量并在前端选择 `provider=tablegpt`。

## 6. Research & Evaluation Ideas
- 比较传统流程（需先跑指标分析）与新上传接口在训练准备时间上的差异。
- 对比有无 R² 展示/SHAP 解释的模型训练周期及最终可用性，展示界面对业务用户的感知提升。
- 量化 TableGPT 在未启动缺失字段校验前后的回答准确率与用户满意度。

## 7. Future Work
- 引入自动特征选择与队列训练调度模块，允许批量试验不同模型配置；
- 将趋势/相关性分析结果回馈 TableGPT，让智能问答自动聚焦当前指标并给出更具操作性的建议；
- 结合真实铝厂生产数据与知识图谱扩展（例如加入化学反应阶段建模与设备运行日志），打造工业级智能决策平台。
<img width="3692" height="1837" alt="屏幕截图 2025-12-24 164257" src="https://github.com/user-attachments/assets/79dc575e-710c-4aaf-9092-7640bf48da22" />
<img width="3598" height="1635" alt="屏幕截图 2025-12-24 165526" src="https://github.com/user-attachments/assets/782caf19-a1b2-4bee-8f8c-cc6e9410ffc4" />
<img width="3749" height="1865" alt="屏幕截图 2025-12-24 165506" src="https://github.com/user-attachments/assets/6944d2a5-8fce-45dd-aa25-18d5adcc7894" />
<img width="3696" height="1874" alt="屏幕截图 2025-12-24 164312" src="https://github.com/user-attachments/assets/6f18b25b-ceda-42bb-a5ac-7bbe9ad596e4" />

<img width="2009" height="1869" alt="屏幕截图 2025-12-24 164541" src="https://github.com/user-attachments/assets/d2feb12e-4e19-4675-ad7f-1716223300ee" />

<img width="3720" height="1828" alt="屏幕截图 2025-12-24 164838" src="https://github.com/user-attachments/assets/5a009135-2a67-4108-82d0-a6b9384991fe" />

<img width="3703" height="1902" alt="屏幕截图 2025-12-24 165337" src="https://github.com/user-attachments/assets/54690fbd-3934-4885-a3b1-f8503abefda3" />


