# 史料RAG

![史料RAG演示](./demo.gif)

本项目展示如何使用[向量数据库](https://zilliz.com.cn/)基于[RAG(检索增强生成)](https://zhuanlan.zhihu.com/p/643953182)方式搭建一个中国历史问答应用。这个应用接受用户的询问，从历史语料库中检索相关的历史资料片段，利用大语言模型给出较为可靠的回答。相比于直接询问大模型，这种方式具有回答准确率高，不容易产生大模型的“幻觉”问题等优点。

本项目实现了两种使用方式，“Milvus方案“在本地启动一个Milvus向量数据库的Docker服务，使用LlamaIndex框架和本地`BAAI/bge-base-zh-v1.5`Embedding模型实现RAG的业务逻辑。“Zilliz Cloud Pipelines方案”使用云上的知识库检索服务Zilliz Cloud Pipelines，该服务包括了RAG中文档切片、向量化、向量检索等功能。两种方案均使用OpenAI的GPT作为大语言模型。

## 运行本项目将会需要:
    拥有OpenAI API Key 
    安装Milvus==2.3.3 或者 拥有Zilliz Cloud账号
    安装LlamaIndex==0.9.22
    安装Docker
    安装python3

## Milvus方案
    
### 步骤1: 配置OpenAI API key

项目中使用OpenAI的GPT作为大语言模型，在开始之前，配置环境变量存放 OpenAI API Key (格式类似于sk-xxxxxxxx)。如果没有，请参考[OpenAI官方文档](https://platform.openai.com/docs/quickstart?context=curl)获取。在terminal中输入以下命令添加环境变量：
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 步骤2: 安装Milvus
使用Docker启动向量数据库Milvus服务，使用的默认端口为19530。如果你使用Mac系统，执行以下命令前请确保Docker Desktop已经安装并运行（如何安装参考[这里](https://dockerdocs.cn/docker-for-mac/install/)）：
```bash
cd db
sudo docker compose up -d
cd ..
```

### 步骤3: 安装Python依赖项
如果你的环境中没有Python3，可以参考[这里](https://www.w3cschool.cn/python3/python3-install.html)安装。

(**可选**)本项目中使用的python依赖可能会和你的现有环境产生冲突，如果你担心这一点，可以使用[`virtualenv`](https://zhuanlan.zhihu.com/p/60647332)工具建立一个新的依赖环境，退出该环境时使用`deactivate`。请注意使用这种方式会重新下载pytorch等依赖项（即便本级已经安装了它们），可能耗时较长。
```bash
pip install virtualenv
virtualenv rag
source rag/bin/activate
```
现在安装所需依赖（以下命令无论是否在virtualenv中都是一样的）：
```bash
pip install -r requirements.txt
```

### 步骤4: 构建史料知识库
利用文本史料构建方便进行RAG的向量索引。执行交互程序cli.py,选择`milvus`模式，然后输入要构建的语料，`build ./data/history_24/`会将该目录下所有文件进行索引构建，耗费时间较长，针对大规模语料库建议使用下面的“Zilliz Cloud Pipelines方案”。
```bash
python cli.py
(rag) milvus
(rag) build ./data/history_24/baihuasanguozhi.txt
```

### 步骤5: 进行问题查询
输入`ask`进入提问模式。输入我们感兴趣的问题。
```bash
(rag) ask
(rag) 问题:关公刮骨疗毒是真的吗
```

## Zilliz Cloud Pipelines方案
    
### 步骤1: 配置OpenAI API key

项目中使用OpenAI的GPT作为大语言模型，在开始之前，配置环境变量存放 OpenAI API Key (格式类似于sk-xxxxxxxx)。如果没有，请参考[OpenAI官方文档](https://platform.openai.com/docs/quickstart?context=curl)获取。在terminal中输入以下命令添加环境变量：
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 步骤2: 获取Zilliz Cloud的配置信息

注册Zilliz Cloud账号，获取相应的配置，这个方案可以利用云端的算力进行大量文档的处理。你可以参考[这里](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/zilliz_pipeline_rag.ipynb)了解更加详细的使用教程。
![Pipeline中所需要的两个配置信息](https://raw.githubusercontent.com/milvus-io/bootcamp/6706a04e45018312905ccb7ad34def031d6937f7/images/zilliz_api_key_cluster_id.jpeg)
同样在环境变量中添加
```bash
export ZILLIZ_TOKEN=<左边红框的信息> 
export ZILLIZ_CLUSTER_ID=<右边红框的信息>
```

### 步骤3: 安装Python依赖项
如果你的环境中没有Python3，可以参考[这里](https://www.w3cschool.cn/python3/python3-install.html)安装。

(**可选**)本项目中使用的python依赖可能会和你的现有环境产生冲突，如果你担心这一点，可以使用[`virtualenv`](https://zhuanlan.zhihu.com/p/60647332)工具建立一个新的依赖环境，退出该环境时使用`deactivate`。请注意使用这种方式会重新下载pytorch等依赖项（即便本级已经安装了它们），可能耗时较长。
```bash
pip install virtualenv
virtualenv rag
source rag/bin/activate
```
现在安装所需依赖
```bash
pip install -r requirements.txt
```

### 步骤4: 构建史料知识库
利用文本史料构建方便进行RAG的向量索引。执行交互程序cli.py,选择`pipeline`模式，然后输入要构建的史料，`build https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/`会将该目录下所有文件进行索引构建, Pipeline方案目前仅支持Url，不支持本地文件。
```bash
python cli.py
(rag) pipeline
(rag) build https://raw.githubusercontent.com/wxywb/history_rag/master/data/history_24/baihuasanguozhi.txt 
```

### 步骤5: 进行问题查询
输入`ask`进入提问模式。输入我们感兴趣的问题。
```bash
ask
问题:关公刮骨疗毒是真的吗
```

## FAQ

**问题**：如何使用别的embedding模型以及reranker模型？
```bash
回答：在config.yaml中进行修改，注意要填写正确的模型向量维度。
```
**问题**：huggingface无法连接上，无法下载模型怎么办？
```bash
回答：将`export HF_ENDPOINT=https://hf-mirror.com`添加到你的环境变量中。
```
**问题**：模型太大，网络连接不稳定，容易失败怎么办？
```bash
回答：以此命令为例，使用`huggingface-cli download --resume-download --local-dir-use-symlinks False BAAI/bge-reranker-large --local-dir bge-reranker-large`
将模型下载到本地,然后就可以进行使用。
```
**问题**：可以添加别的史料吗？
```bash
回答：可以，但是由于会根据纪传体格式来判断引用时候的章节名，所以最好是每一个章节以"某某传"开头(无缩进)，然后使用缩进来表示正文。
```
**问题**：可以使用其他LLM吗？
```bash
回答：可以，Llama Index所支持的LLM都可以很轻松的使用, 由于默认使用的是OpenAI的模型，所以需要在`executor.py`中初始化其他LLM来进行集成。
```

## 指令附录

| Command                   | Flag(s)          | Description                                      |
|---------------------------|------------------|--------------------------------------------------|
| build                   |                 | 将`目录`或者`文件`进行索引构建，Milvus模式下为本地文件，Pipeline下为Url                    |
| build                  | -overwrite       | 同上，但是进行覆盖构建，已有索引将被清空       |
| ask                     |                 | 进入问答模式，输入`quit`退出该模式             |
| ask                     | -d               | 同上，但是开启Debug模式，将返回出来搜索出来的语料信息    |
| remove                  |                |  删除「文件名」中倒入的索引                     |
| quit                    |                 | 退出当前状态                                     |





