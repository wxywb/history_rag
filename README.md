# 史料RAG

本项目展示如何使用Llama Index与Milvus搭建一个中国历史知识的RAG(检索增强生成系统)系统,这种系统可以根据询问来从历史语料库中检索到相关的历史资料片段，从而使模型给出更可靠的回答。

## 依赖:
    OpenAI token
    Milvus==2.3.3
    LlamaIndex==0.9.22
    Docker
    python3

## 步骤
    
### 步骤1: 获取OpenAI Token

在开始之前，确保你有有效的OpenAI token。如果没有，请参考OpenAI的文档获取。并在环境变量中添加
```bash
export OPENAI_API_KEY=sk-xxxxxxxx你的tokenxxxxxxxxx
```

### 步骤2: 安装milvus
启动向量数据库Milvus
```bash
cd db
sudo docker compose up -d
```

### 步骤3: 安装依赖
```bash
pip install -r requirements.txt
```

### 步骤4: 构建历史知识库
利用文本语料库构建方便进行RAG的向量索引。分别为`sanguo`(三国志)和`history24`(24史),如果你有自己的语料，可以在common.py中添加
```bash
python build_index.py --corpus sanguo
```


### 步骤5: 进行问题查询
打开query.py,确保你查询的是正确的索引。我们想在三国志语料上进行查询
```python
    dataset_name = 'sanguo'
```
输入我们感兴趣的问题。
```python
    query = '关公刮骨疗毒是真的吗'
```
最后执行查看输出。
```bash
python query.py
```
**提示**:如果想查看搜索结果，将`common.py`中的`retrieve_debug`设置为`True`。

