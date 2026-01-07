# Demo开发与实验指南 - AI芯片应用开发岗位

```infographic
infographic list-grid-badge-card
data
  title 三大Demo方向
  items
    - label 大模型部署优化
      desc 岗位核心技能展示
      icon mdi:chip
    - label 飞书AI应用
      desc 岗位明确要求
      icon mdi:robot
    - label Agent Sandbox
      desc 创新项目亮点
      icon mdi:laboratory
```

---

## 📋 Demo 1: 大模型部署优化

### 项目概述

**名称**：ChatGLM-6B量化部署与性能优化

**目标**：
- 在有限GPU资源下部署大模型
- 应用多种优化技术
- 生成详细性能测试报告

**时间安排**：第8-10周（3周）

---

### 技术路线

```infographic
infographic sequence-steps-simple
data
  title 开发流程
  items
    - label 环境准备
      desc 安装依赖、下载模型
    - label 基线测试
      desc FP16精度性能基准
    - label 量化优化
      desc INT8/INT4量化
    - label 推理加速
      desc vLLM/Flash Attention
    - label 性能对比
      desc 生成测试报告
```

---

### 实现步骤

#### 步骤1: 环境搭建（Day 1-2）

```bash
# 创建项目目录
mkdir llm-deployment-demo && cd llm-deployment-demo

# 创建虚拟环境
conda create -n llm-deploy python=3.10
conda activate llm-deploy

# 安装依赖
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.0  # 量化库
pip install vllm==0.2.6  # 推理加速
pip install fastapi uvicorn  # API服务
pip install locust  # 压力测试
```

**项目结构**：
```
llm-deployment-demo/
├── models/               # 模型权重
├── src/
│   ├── inference.py     # 推理脚本
│   ├── quantize.py      # 量化脚本
│   ├── benchmark.py     # 性能测试
│   └── api_server.py    # FastAPI服务
├── configs/
│   └── model_config.yaml
├── benchmarks/
│   └── results/         # 测试结果
├── notebooks/
│   └── analysis.ipynb   # 结果分析
├── requirements.txt
└── README.md
```

---

#### 步骤2: 模型下载与基线测试（Day 3-4）

```python
# src/inference.py
from transformers import AutoTokenizer, AutoModel
import torch
import time

class BaselineInference:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16  # FP16基线
        ).cuda()
        self.model.eval()
    
    def generate(self, prompt, max_length=512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7
            )
        latency = time.time() - start_time
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, latency
    
    def get_memory_usage(self):
        """获取显存占用"""
        return torch.cuda.max_memory_allocated() / 1024**3  # GB

# 测试脚本
if __name__ == "__main__":
    engine = BaselineInference()
    
    test_prompts = [
        "解释什么是Transformer",
        "用Python写一个快速排序",
        "AI芯片的主要类型有哪些？"
    ]
    
    print("=== FP16 Baseline ===")
    for prompt in test_prompts:
        response, latency = engine.generate(prompt)
        print(f"Prompt: {prompt}")
        print(f"Latency: {latency:.2f}s")
        print(f"Memory: {engine.get_memory_usage():.2f}GB\n")
```

---

#### 步骤3: INT8量化（Day 5-7）

```python
# src/quantize.py
from transformers import AutoTokenizer, AutoModel
import torch

class QuantizedInference:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        """使用bitsandbytes进行INT8量化"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # 加载INT8量化模型
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_8bit=True,  # INT8量化
            device_map="auto"
        )
        self.model.eval()
    
    def generate(self, prompt, max_length=512):
        # 同上，推理代码
        pass

# 对比测试
if __name__ == "__main__":
    print("Loading INT8 model...")
    int8_engine = QuantizedInference()
    
    # 运行相同测试
    # ...
```

---

#### 步骤4: vLLM推理加速（Day 8-10）

```python
# src/vllm_inference.py
from vllm import LLM, SamplingParams

class VLLMInference:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # 单GPU
            dtype="float16",
            max_model_len=2048
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )
    
    def generate_batch(self, prompts):
        """批量推理"""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

# 批量测试（体现吞吐量优势）
if __name__ == "__main__":
    engine = VLLMInference()
    
    # 批量请求
    batch_prompts = ["问题1", "问题2", "问题3"] * 10  # 30个请求
    
    start = time.time()
    results = engine.generate_batch(batch_prompts)
    total_time = time.time() - start
    
    print(f"Throughput: {len(batch_prompts) / total_time:.2f} req/s")
```

---

#### 步骤5: API服务与压力测试（Day 11-14）

```python
# src/api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm_inference import VLLMInference
import uvicorn

app = FastAPI(title="LLM Inference API")
engine = VLLMInference()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

class GenerateResponse(BaseModel):
    text: str
    latency: float

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    import time
    start = time.time()
    
    result = engine.generate_batch([request.prompt])[0]
    latency = time.time() - start
    
    return GenerateResponse(text=result, latency=latency)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**压力测试**（Locust）：
```python
# benchmarks/locustfile.py
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def generate_text(self):
        self.client.post("/generate", json={
            "prompt": "什么是深度学习？",
            "max_tokens": 256
        })

# 运行: locust -f locustfile.py --host=http://localhost:8000
```

---

#### 步骤6: 性能测试与报告（Day 15-21）

**自动化benchmark脚本**：
```python
# src/benchmark.py
import json
import matplotlib.pyplot as plt
import pandas as pd
from baseline_inference import BaselineInference
from quantized_inference import QuantizedInference
from vllm_inference import VLLMInference

def benchmark_all():
    engines = {
        "FP16 Baseline": BaselineInference(),
        "INT8 Quantized": QuantizedInference(),
        "vLLM FP16": VLLMInference()
    }
    
    test_prompts = [
        "短提示测试",
        "中等长度的提示" * 10,
        "很长的提示" * 50
    ]
    
    results = []
    
    for name, engine in engines.items():
        print(f"Testing {name}...")
        for prompt_type, prompt in enumerate(test_prompts):
            latency, memory = engine.test(prompt)
            results.append({
                'Engine': name,
                'PromptType': f"Type{prompt_type+1}",
                'Latency(s)': latency,
                'Memory(GB)': memory
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv('benchmarks/results/comparison.csv', index=False)
    
    # 可视化
    plot_results(df)
    
    return df

def plot_results(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 延迟对比
    df.pivot(index='PromptType', columns='Engine', values='Latency(s)').plot(
        kind='bar', ax=axes[0], title='Latency Comparison'
    )
    axes[0].set_ylabel('Latency (seconds)')
    
    # 内存对比
    df.groupby('Engine')['Memory(GB)'].mean().plot(
        kind='bar', ax=axes[1], title='Memory Usage'
    )
    axes[1].set_ylabel('Memory (GB)')
    
    plt.tight_layout()
    plt.savefig('benchmarks/results/comparison.png', dpi=300)
    print("Results saved to benchmarks/results/")

if __name__ == "__main__":
    benchmark_all()
```

**生成技术报告**：
```markdown
# ChatGLM-6B 部署优化报告

## 实验环境
- GPU: NVIDIA RTX 3090 (24GB)
- CUDA: 11.8
- PyTorch: 2.1.0

## 优化方法对比

| 方法 | 延迟 (s) | 显存 (GB) | 吞吐量 (tok/s) | 精度损失 |
|------|----------|-----------|----------------|----------|
| FP16 Baseline | 2.34 | 13.2 | 45 | - |
| INT8 Quantized | 1.98 | 7.1 | 52 | <1% |
| vLLM+FP16 | 0.87 | 13.5 | 118 | - |
| vLLM+INT8 | 0.76 | 7.3 | 135 | <1% |

## 关键发现
1. **vLLM带来3x吞吐量提升**（PagedAttention）
2. **INT8量化节省46%显存**，性能损失可忽略
3. **组合优化效果最佳**：vLLM+INT8

## 优化技术详解
### 1. INT8量化
- 使用LLM.int8()算法
- 混合精度：敏感层保持FP16
- 实现细节：...

### 2. vLLM优化
- PagedAttention减少显存碎片
- Continuous batching提升吞吐
- ...

## 结论
通过量化和推理优化，在保持精度的前提下：
- ✅ 显存占用减少46%
- ✅ 推理速度提升3倍
- ✅ 可支持更大batch size
```

---

### Demo展示方式

#### 方式1: Jupyter Notebook交互式展示

**创建**：`notebooks/demo.ipynb`

**内容结构**：
1. **问题引入**：大模型部署挑战
2. **方案对比**：运行不同优化方法
3. **实时可视化**：显存占用、推理速度
4. **结论总结**：性能提升数据

#### 方式2: Gradio Web界面

```python
# demo_app.py
import gradio as gr
from vllm_inference import VLLMInference

engine = VLLMInference()

def generate_text(prompt, method):
    """
    method: 'FP16', 'INT8', 'vLLM'
    """
    # 根据method选择不同引擎
    result = engine.generate([prompt])[0]
    return result

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="输入提示", placeholder="请输入问题..."),
        gr.Dropdown(["FP16 Baseline", "INT8", "vLLM"], label="优化方法")
    ],
    outputs=gr.Textbox(label="模型输出"),
    title="大模型部署优化Demo",
    description="对比不同优化方法的效果"
)

demo.launch()
```

#### 方式3: 录制演示视频（5分钟）

**脚本**：
1. **0:00-0:30** - 介绍背景（大模型部署挑战）
2. **0:30-1:30** - 展示代码结构（快速浏览）
3. **1:30-3:00** - 运行benchmark（屏幕录制）
4. **3:00-4:00** - 结果可视化（图表讲解）
5. **4:00-5:00** - 总结与技术要点

**工具**：OBS Studio录屏

---

## 📱 Demo 2: 飞书AI智能助手

### 项目概述

**名称**：飞书知识库问答机器人

**核心功能**：
1. 接收飞书消息
2. 检索知识库（RAG）
3. 调用本地LLM生成回答
4. 返回飞书

**时间安排**：第13-15周（3周）

---

### 系统架构

```infographic
infographic hierarchy-tree-curved-line-rounded-rect-node
data
  title 飞书AI助手架构
  items
    - label 飞书客户端
      children:
        - label 用户发送消息
    - label 飞书开放平台
      children:
        - label Webhook回调
        - label 消息API
    - label 后端服务 (FastAPI)
      children:
        - label RAG检索模块
        - label LLM推理模块
        - label 对话管理
    - label 数据层
      children:
        - label 向量数据库
        - label 对话历史
```

---

### 实现步骤

#### 步骤1: 飞书应用创建（Day 1-2）

**操作流程**：
1. 访问 https://open.feishu.cn/
2. 创建企业自建应用
3. 获取 App ID 和 App Secret
4. 配置权限：
   - 读取消息
   - 发送消息
   - 获取用户信息
5. 配置事件订阅URL（后续填写）

**配置文件**：
```yaml
# config/feishu_config.yaml
app_id: "cli_xxxxx"
app_secret: "xxxxxx"
verification_token: "xxxxxx"
encrypt_key: "xxxxxx"  # 可选

webhook_url: "https://your-server.com/feishu/webhook"
```

---

#### 步骤2: 后端服务搭建（Day 3-7）

```python
# src/feishu_bot.py
from fastapi import FastAPI, Request, HTTPException
from lark_oapi.api.im.v1 import *
import lark_oapi as lark
import os

app = FastAPI()

# 初始化飞书客户端
client = lark.Client.builder() \
    .app_id(os.getenv("FEISHU_APP_ID")) \
    .app_secret(os.getenv("FEISHU_APP_SECRET")) \
    .build()

@app.post("/feishu/webhook")
async def feishu_webhook(request: Request):
    """接收飞书事件回调"""
    body = await request.json()
    
    # 验证challenge
    if "challenge" in body:
        return {"challenge": body["challenge"]}
    
    # 处理消息事件
    if body.get("header", {}).get("event_type") == "im.message.receive_v1":
        await handle_message(body)
    
    return {"code": 0}

async def handle_message(event_data):
    """处理接收到的消息"""
    message = event_data["event"]["message"]
    content = json.loads(message["content"])
    user_input = content.get("text", "")
    
    # 调用AI生成回复
    ai_response = await generate_response(user_input)
    
    # 发送回复到飞书
    await send_message(message["chat_id"], ai_response)

async def send_message(chat_id, text):
    """发送消息到飞书"""
    request = CreateMessageRequest.builder() \
        .receive_id_type("chat_id") \
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("text")
            .content(json.dumps({"text": text}))
            .build()
        ).build()
    
    response = client.im.v1.message.create(request)
    
    if not response.success():
        print(f"Error: {response.msg}")

# AI生成逻辑（下一步实现）
async def generate_response(user_input):
    # TODO: 集成LLM
    return "收到：" + user_input
```

---

#### 步骤3: RAG知识库集成（Day 8-12）

```python
# src/rag_engine.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

class RAGEngine:
    def __init__(self, knowledge_base_path="./knowledge_base"):
        # 加载文档
        loader = DirectoryLoader(knowledge_base_path, glob="**/*.md")
        documents = loader.load()
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)
        
        # 创建向量数据库
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5"  # 中文embedding
        )
        self.vectorstore = FAISS.from_documents(texts, embeddings)
    
    def retrieve(self, query, top_k=3):
        """检索相关文档"""
        docs = self.vectorstore.similarity_search(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

# 集成到消息处理
rag_engine = RAGEngine()

async def generate_response(user_input):
    # 检索相关知识
    context = rag_engine.retrieve(user_input)
    
    # 构建prompt
    prompt = f"""根据以下知识库内容回答问题：

知识库：
{context}

问题：{user_input}

回答："""
    
    # 调用LLM（使用Demo1中的推理引擎）
    from vllm_inference import VLLMInference
    llm = VLLMInference()
    response = llm.generate_batch([prompt])[0]
    
    return response
```

---

#### 步骤4: 对话历史管理（Day 13-15）

```python
# src/conversation_manager.py
from collections import defaultdict
import json

class ConversationManager:
    def __init__(self, max_history=5):
        self.conversations = defaultdict(list)
        self.max_history = max_history
    
    def add_message(self, chat_id, role, content):
        """添加消息到历史"""
        self.conversations[chat_id].append({
            "role": role,
            "content": content
        })
        
        # 保持最近N轮对话
        if len(self.conversations[chat_id]) > self.max_history * 2:
            self.conversations[chat_id] = self.conversations[chat_id][-self.max_history*2:]
    
    def get_history(self, chat_id):
        """获取对话历史"""
        return self.conversations[chat_id]
    
    def format_prompt(self, chat_id, current_query, context=""):
        """格式化为模型输入"""
        history = self.get_history(chat_id)
        
        prompt = f"你是一个智能助手。\n\n"
        
        if context:
            prompt += f"参考信息：\n{context}\n\n"
        
        prompt += "对话历史：\n"
        for msg in history:
            prompt += f"{msg['role']}: {msg['content']}\n"
        
        prompt += f"用户: {current_query}\n助手: "
        
        return prompt

# 更新generate_response
conv_manager = ConversationManager()

async def generate_response(user_input, chat_id):
    # 检索知识
    context = rag_engine.retrieve(user_input)
    
    # 构建带历史的prompt
    prompt = conv_manager.format_prompt(chat_id, user_input, context)
    
    # 生成回复
    response = llm.generate_batch([prompt])[0]
    
    # 保存对话
    conv_manager.add_message(chat_id, "用户", user_input)
    conv_manager.add_message(chat_id, "助手", response)
    
    return response
```

---

#### 步骤5: 部署与测试（Day 16-21）

**Docker部署**：
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY knowledge_base/ ./knowledge_base/

EXPOSE 8000

CMD ["uvicorn", "src.feishu_bot:app", "--host", "0.0.0.0", "--port", "8000"]
```

**内网穿透测试**（开发阶段）：
```bash
# 使用ngrok暴露本地服务
ngrok http 8000

# 将生成的URL配置到飞书应用的事件订阅地址
```

---

### Demo展示

#### 展示脚本

1. **问题演示**：在飞书中发送问题
   - "公司的AI芯片支持哪些框架？"
   - "如何部署大模型？"

2. **后台展示**：
   - 终端显示接收到消息
   - RAG检索日志
   - LLM生成过程

3. **结果展示**：飞书中收到AI回复

4. **技术讲解**：
   - RAG检索机制
   - 对话历史管理
   - 飞书API集成

---

## 🤖 Demo 3: Agent Sandbox 原型

### 快速原型（MVP）

**时间**：第16周（1周）

**目标**：证明概念可行性

```python
# sandbox_mvp.py
import gymnasium as gym
import torch
from stable_baselines3 import PPO
import time

class ChipAwarePPO:
    """简化版芯片感知训练"""
    
    def __init__(self, env_name="CartPole-v1", device="cuda"):
        self.env = gym.make(env_name)
        self.device = device
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            device=device,
            verbose=1,
            tensorboard_log="./logs/"
        )
    
    def train_with_monitoring(self, total_timesteps=10000):
        """训练并监控硬件"""
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        callback = GPUMonitorCallback(handle)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
    
    def evaluate(self):
        """评估Agent"""
        obs, _ = self.env.reset()
        total_reward = 0
        
        for _ in range(500):
            action, _ = self.model.predict(obs)
            obs, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        return total_reward

from stable_baselines3.common.callbacks import BaseCallback

class GPUMonitorCallback(BaseCallback):
    def __init__(self, gpu_handle):
        super().__init__()
        self.gpu_handle = gpu_handle
    
    def _on_step(self):
        if self.n_calls % 100 == 0:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            print(f"Step {self.n_calls}: GPU Util = {util.gpu}%")
        return True

# 运行Demo
if __name__ == "__main__":
    agent = ChipAwarePPO()
    
    print("开始训练...")
    agent.train_with_monitoring(total_timesteps=50000)
    
    print("评估Agent...")
    reward = agent.evaluate()
    print(f"总奖励: {reward}")
```

**展示要点**：
- 实时显示训练过程
- TensorBoard可视化
- GPU利用率监控
- 对比不同设备（CPU vs GPU）


### 1. 大模型部署优化
- 实现ChatGLM-6B的INT8量化，显存占用减少46%
- 使用vLLM优化，推理速度提升3倍
- [查看详情](./projects/llm-deployment/) | [GitHub](https://github.com/你的用户名/llm-deployment)

### 2. 飞书AI智能助手
- 基于RAG的知识库问答系统
- 支持上下文对话
- [在线Demo](链接) | [GitHub](...)

### 3. Agent Sandbox原型
- 硬件感知的强化学习环境
- 支持NVIDIA/AMD多芯片适配
- [技术文档](链接) | [GitHub](...)

## 技能矩阵
- Python, C++, CUDA
- PyTorch, Transformers, vLLM
- 飞书开放平台开发
- 模型量化与优化

## 🧪 实验记录规范

### 实验日志模板

```markdown
# 实验日志 - [日期]

## 实验目标
明确本次实验要验证什么

## 实验配置
- 硬件：GPU型号、内存
- 软件：框架版本
- 模型：模型名称、参数量

## 实验步骤
1. ...
2. ...

## 实验结果
### 定量结果
| 指标 | 值 |
|------|---|
| ... | ... |

### 定性观察
- 现象1
- 现象2

## 问题与解决
- **问题**：CUDA out of memory
  - **解决**：减小batch size到16

---

## ✅ 最终检查清单

### Demo 1: 大模型部署
- [ ] 代码运行无误
- [ ] 生成性能对比报告
- [ ] 可视化图表清晰
- [ ] README文档完整
- [ ] 录制演示视频

### Demo 2: 飞书AI助手
- [ ] 飞书应用配置正确
- [ ] RAG检索功能正常
- [ ] 对话历史管理生效
- [ ] 部署文档详细
- [ ] 准备测试对话案例

### Demo 3: Agent Sandbox
- [ ] MVP原型可运行
- [ ] GPU监控功能正常
- [ ] TensorBoard可视化
- [ ] 架构文档撰写
- [ ] 未来规划清晰
---
## 📚 资源链接汇总
### 官方文档
- 飞书开放平台: https://open.feishu.cn/document/
- vLLM文档: https://docs.vllm.ai/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

