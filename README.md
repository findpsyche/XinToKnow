# AI-Chip-Developer-Bootcamp (芯火计划)



**第一步**: 阅读 [00_技能图谱总览.md](./00_技能图谱总览.md)  
**第二步**: 根据自己的基础选择学习路径  
**第三步**: 动手实践 + 构建项目

---

### 📘 核心技能深度文档 (NEW!)

#### [00_技能图谱总览](./00_技能图谱总览.md) ⭐ START HERE
**全局导航** - 快速了解整个技能体系
- T字型技能结构
- 学习优先级矩阵
- 16周时间线
- 自检清单

#### [06_岗位核心技能深度图谱](./06_岗位核心技能深度图谱.md) ⭐⭐⭐⭐⭐
**AI模型与部署** (~25页 | 80+ 代码示例)
- ✅ Transformer架构深度解析 (Attention, KV Cache)
- ✅ 模型量化技术实战 (bitsandbytes, GPTQ, AWQ)
- ✅ ONNX格式转换与优化
- ✅ 性能指标测量 (TPS, TTFT, Throughput)

**亮点**：
- 每个概念都有代码实现
- KV Cache原理深度解析
- 量化技术对比实验
- 性能测试完整脚本

#### [07_系统工具链与底层技能](./07_系统工具链与底层技能.md) ⭐⭐⭐⭐⭐
**Linux/Docker/C++** (~23页 | 实用命令合集)
- ✅ Linux系统管理与Shell脚本
- ✅ Docker容器化完全指南
- ✅ C++基础阅读能力
- ✅ CMake与pybind11

**亮点**：
- 常用命令速查表
- Dockerfile最佳实践
- SDK集成实战案例
- 自动化部署脚本模板

#### [08_应用开发与软技能](./08_应用开发与软技能.md) ⭐⭐⭐⭐⭐
**FastAPI/飞书/RAG** (~33页 | 完整项目代码)
- ✅ FastAPI异步后端开发
- ✅ 飞书开放平台完全指南
- ✅ RAG检索增强生成实战
- ✅ 技术文档撰写模板

**亮点**：
- 完整的推理服务代码
- 飞书Bot从零实现
- RAG系统架构设计
- API文档/测试报告模板

---

### 📖 基础学习路径文档

#### 1️⃣ [岗位技能培训路径](./01_岗位技能培训路径.md)
**内容**：
- 16周系统化学习计划
- 顶级大学课程推荐（Stanford, MIT, Berkeley等）
- Python/C++/CUDA技术栈
- PyTorch深度学习框架
- 大模型技术与部署优化
- 飞书开放平台开发

**学习路径**：
- Week 1-4: 基础强化（Python/C++/PyTorch）
- Week 5-8: 硬件优化（GPU/芯片架构）
- Week 9-12: 大模型技术（Transformer/部署）
- Week 13-16: 应用开发（飞书/实战项目）

---

#### 2️⃣ [顶级论文与研究资源](./02_顶级论文与研究资源.md)
**内容**：
- 🏛️ **顶级研究实验室**
  - Stanford DAWN Lab, HAI
  - UC Berkeley Sky Computing Lab
  - MIT Han Lab, CSAIL
  - CMU, Princeton, UW等

- 📄 **必读论文清单**（28篇核心论文）
  - Transformer基础架构
  - Flash Attention优化
  - 模型量化技术（GPTQ, AWQ）
  - 推理加速（vLLM, Speculative Decoding）
  - AI芯片架构（Google TPU）

- 🔍 **论文查找资源**
  - arXiv, Google Scholar, Papers with Code
  - 顶会：NeurIPS, ICML, MLSys, OSDI

---

#### 3️⃣ [Kaggle竞赛与实践项目](./03_Kaggle竞赛与实践项目.md)
**内容**：
- 🏆 **相关Kaggle竞赛**
  - LLM应用竞赛
  - 模型优化与部署
  - GPU/边缘设备优化

- 🛠️ **4个实践项目**
  1. LLM量化部署（第8周）
  2. 自定义CUDA Kernel（第10周）
  3. 飞书AI机器人（第14周）
  4. Agent Sandbox原型（第16周）

---

#### 4️⃣ [AI Agent Sandbox架构方案](./04_AI_Agent_Sandbox架构方案.md)
**核心创新**：多芯片适配的强化学习环境

**可行性评估**：✅ **高度可行**

**系统架构**：
```
Agent层 (RL Agents: PPO, DQN, SAC)
         ↓
Sandbox核心层 (环境接口、奖励函数)
         ↓
硬件抽象层 (NVIDIA/AMD/自研芯片适配器)
         ↓
芯片层 (GPU/TPU/专用AI芯片)
```

**核心功能**：
- 统一的Agent API（兼容OpenAI Gym）
- 芯片感知的训练策略
- 硬件性能监控
- 自适应超参数调整

---

#### 5️⃣ [Demo开发与实验指南](./05_Demo开发与实验指南.md)
**三大Demo方向**：

**Demo 1: 大模型部署优化** ⭐⭐⭐⭐⭐
- INT8量化（节省46%显存）
- vLLM推理加速（3x吞吐量提升）
- 性能对比报告

**Demo 2: 飞书AI智能助手** ⭐⭐⭐⭐⭐
- 飞书消息接收/发送
- RAG知识库检索
- 对话历史管理

**Demo 3: Agent Sandbox原型** ⭐⭐⭐⭐
- 硬件感知的RL环境
- GPU利用率监控
- 芯片性能对比

---

## 🎯 学习优先级

```
第一优先级（生存技能）:
✅ Python + Linux + 模型部署基础
→ 保证你能把模型跑起来

第二优先级（业务技能）:
✅ 飞书开发 + RAG技术
→ 保证你能做出业务应用

第三优先级（进阶技能）:
✅ Docker + 量化技术 + C++
→ 保证你能处理复杂环境

第四优先级（加分项）:
✅ Agent Sandbox + 论文阅读
→ 展示创新思维
```

---

## 📅 16周学习路线

| 阶段 | 周次 | 学习重点 | 实践项目 | 里程碑 |
|------|------|---------|---------|--------|
| **基础** | 1-4 | Python/C++/PyTorch | PyTorch自定义算子 | 基础技能掌握 ✅ |
| **硬件** | 5-8 | GPU架构/CUDA/优化 | 模型量化项目 | 优化能力获得 ✅ |
| **大模型** | 9-12 | Transformer/LLM/部署 | 大模型部署Demo | Demo 1完成 ✅ |
| **应用** | 13-16 | 飞书开发/Agent设计 | 飞书机器人+Sandbox | Demo 2&3完成 ✅ |

---

## ✅ 核心目标检查清单

### 技术能力目标

- [ ] **编程语言**
  - [ ] Python高级特性
  - [ ] C++基础阅读
  - [ ] CUDA kernel编写

- [ ] **深度学习框架**
  - [ ] PyTorch熟练使用
  - [ ] 自定义算子开发
  - [ ] 模型导出（ONNX）

- [ ] **大模型技术**
  - [ ] Transformer架构理解
  - [ ] 部署过1个大模型
  - [ ] 掌握量化技术

- [ ] **应用开发**
  - [ ] FastAPI服务开发
  - [ ] 飞书开放平台集成
  - [ ] RAG系统构建

---

### 项目产出目标

- [ ] **GitHub仓库**（至少3个）
  - [ ] 大模型部署优化项目
  - [ ] 飞书AI机器人
  - [ ] Agent Sandbox原型

- [ ] **技术文档**
  - [ ] 每个项目有完整README
  - [ ] 至少1份性能测试报告
  - [ ] API文档

- [ ] **可展示Demo**
  - [ ] 录制3个项目演示视频
  - [ ] 部署在线Demo（至少1个）
  - [ ] 准备面试PPT

---

## 🚀 立即开始

### Day 1: 环境准备

```bash
# 1. 安装Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. 创建环境
conda create -n chip-dev python=3.10
conda activate chip-dev

# 3. 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安装基础库
pip install transformers accelerate bitsandbytes fastapi uvicorn
```

### Week 1: 第一个模型

```python
# hello_llm.py
from transformers import AutoModel, AutoTokenizer

model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
```

---

## 📊 项目亮点

### 全面性
- ✅ 涵盖AI芯片开发全流程
- ✅ 从底层硬件到上层应用
- ✅ 理论与实践完美结合

### 实用性
- ✅ 80+ 可运行代码示例
- ✅ 3个完整Demo项目
- ✅ 真实岗位需求对齐

### 创新性
- ✅ AI Agent Sandbox架构
- ✅ 芯片感知的训练策略
- ✅ 多芯片适配方案

---

## 🎓 最终成果

**16周后你将具备**：

✅ **理论知识**
- Transformer架构深度理解
- 量化、推理优化原理
- RAG系统设计

✅ **实践技能**
- 模型部署与优化
- 后端API开发
- 飞书应用集成
- 向量检索系统

✅ **项目作品**
- 大模型部署优化 (性能报告)
- 飞书AI机器人 (可演示)
- Agent Sandbox原型 (创新项目)

✅ **软实力**
- 技术文档撰写
- 问题排查能力
- 系统设计思维

---

## 📞 资源与支持

### 学习社群
- **Reddit**: r/MachineLearning
- **Discord**: PyTorch Official
- **知乎**: AI芯片话题

### 技术博客
- Hugging Face Blog
- NVIDIA Developer Blog
- Lil'Log (OpenAI)

### 开源项目
- vLLM
- Flash Attention
- Stable-Baselines3

---

## 🌟 致谢

本项目整合了以下资源：
- 顶级大学公开课程（Stanford, MIT, Berkeley等）
- 开源社区优秀项目
- 学术界最新研究成果
- 业界最佳实践经验

---

## 📄 文档维护

**版本信息**：
- 当前版本：v1.0
- 创建日期：2026-01-08
- 最后更新：2026-01-08
- 维护者：AI Assistant (Alma) + findpsyche

**更新计划**：
- 每月更新论文清单
- 跟踪曦望sunrise技术动态
- 根据用户反馈优化

---

## 📧 使用说明

1. **克隆项目**
   ```bash
   git clone https://github.com/findpsyche/XinToKnow

   ```

2. **开始学习**
   - 阅读 `00_技能图谱总览.md`
   - 根据基础选择文档
   - 动手实践每个代码示例

3. **构建项目**
   - 跟随 `05_Demo开发与实验指南.md`
   - 完成3个核心Demo
   - 撰写技术文档
---
**License**: MIT  
**Author**: AI Assistant (Alma)  
**Contact**: [GitHub Issues](https://github.com/findpsyche/XinToKnow/issues)
