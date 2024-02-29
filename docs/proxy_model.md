# 使用兼容openai接口的大模型   



(对兼容openai接口的大模型服务支持由[darius-gs](https://github.com/darius-gs)提供)  

用法:
1. 以服务的方式运行本地大模型  
1.1 使用常见的框架，例如：  
chatchat:  
https://github.com/chatchat-space/Langchain-Chatchat/wiki/%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E9%83%A8%E7%BD%B2#%E4%B8%80%E9%94%AE%E5%90%AF%E5%8A%A8  
fastchat:  
https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md  
db-gpt:  
https://www.yuque.com/eosphoros/dbgpt-docs/yg5hynslce8nbge1  
1.2 使用模型自带的服务，例如：  
https://github.com/THUDM/ChatGLM2-6B?tab=readme-ov-file#api-%E9%83%A8%E7%BD%B2  
2. 在cfgs/config_proxy_model.yaml中llm下填写name、api_base、api_key
3. 执行python cli.py --cfg cfgs/config_proxy_model.yaml
4. 其他步骤与原教程相同
   
注：理论上只要兼容openai接口的大模型服务就可以，我使用chatchat和chatglm3-6b-32k以milvus模式进行测试无误  

问题：
暂无


