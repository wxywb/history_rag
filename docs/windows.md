# windows下安装事项

### 步骤1: 配置OpenAI API key

项目中使用OpenAI的GPT4作为大语言模型，在开始之前，配置环境变量存放 OpenAI API Key (格式类似于sk-xxxxxxxx)。如果没有，请参考[OpenAI官方文档](https://platform.openai.com/docs/quickstart?context=curl)获取。
- 单击“高级系统设置”。
- 点击“环境变量”按钮。
- 在“系统变量”部分，点击“新建...”，然后输入变量名称为OPENAI_API_KEY，变量值为您的API密钥。
- 验证：要验证设置是否有效，请重新打开命令提示符并键入以下命令。它应该显示您的API密钥：echo %OPENAI_API_KEY%

### 步骤2: 安装Milvus
使用Docker启动向量数据库Milvus服务，使用的默认端口为19530。下载[文件](https://github.com/milvus-io/milvus/releases/download/v2.3.7/milvus-standalone-docker-compose.yml),在cmd中运行
```bash
docker-compose -f milvus-standalone-docker-compose.yml up -d
```

### 步骤3: 安装Python依赖项
如果你的环境中没有Python3，可以参考[这里](https://docs.anaconda.com/free/anaconda/install/windows/)安装。

```bash
pip install -r requirements.txt
```

### 步骤4: 构建史料知识库
导入文本史料构建知识库，该过程中会将文本切片并生成向量，构建向量索引。

执行交互程序cli.py,选择`milvus`模式，然后输入要构建的语料，例如`build ./data/history_24/baihuasanguozhi.txt`会将白话版《三国志》导入。
```bash
python cli.py
(rag) milvus
(rag) build ./data/history_24/baihuasanguozhi.txt
```
注意，二十四史语料库较大。如果输入`build ./data/history_24/`会将该目录下所有文件进行索引构建，耗费时间较长，针对大规模语料库建议使用下面的“Zilliz Cloud Pipelines方案”。

### 步骤5: 进行问题查询
输入`ask`进入提问模式。输入我们感兴趣的问题。
```bash
(rag) ask
(rag) 问题:关公刮骨疗毒是真的吗
```

