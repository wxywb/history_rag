# 使用Google Gemini Pro/Ultra作为大模型

(对Google Gemini Pro/Ultra的支持由[BetterAndBetterII](https://github.com/betterandbetterii)提供)

用法:
1. 申请Google Gemini Pro的API Key [API申请](https://makersuite.google.com/app/apikey) （只需要Google账号即可免费申请和使用）
2. 设置环境变量 GOOGLE_API_KEY="YOUR_KEY"   (可选设置GOOGLE_BASE_URL="中转的API地址")
3. 修改配置文件config.yaml中的`name`为`gemini-pro`或`gemini-ultra`。
4. 其他步骤与原教程相同。(ZillizPipeline方案类似)

(GeminiLLM.py参考QwenLLM.py编写)

由于llama index中的Gemini无法配置transport='rest'，无法使用中转的API地址，所以用GeminiLLM.py手动实现llama的接口并支持修改传输的方法。