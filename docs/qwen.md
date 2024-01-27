# 使用通义千问作为大模型

(对通义千问的支持由[leyiang](https://github.com/leyiang)提供)

用法:
1. 根据[教程](https://help.aliyun.com/zh/dashscope/developer-reference/api-details)安装sdk, 设置key。
2. 执行python cli.py --cfg cfgs/config_qwen.yaml
3. 其他步骤与原教程相同。

问题：
1. 代码好像会对同一个问题多轮提问，以优化回答，API大部分都会回复："新的上下文没什么帮助..."
2.  Qwen 对于部分问题会报错 "DataInspectionFailed"。

样例：
![2024-01-26-232909_2401x231_scrot](https://github.com/wxywb/history_rag/assets/39115827/d49c0550-10f5-4862-9939-697ed9dcc8c9)
![2024-01-26-232953_2418x210_scrot](https://github.com/wxywb/history_rag/assets/39115827/47687e76-8115-4164-af13-c83c5fcff6cb)


