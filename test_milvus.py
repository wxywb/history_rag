from pymilvus import connections
try:
    connections.connect("default", host="127.0.0.1", port="19530", timeout=5)
    print("✅ 成功连上 Milvus 了！")
except Exception as e:
    print(f"❌ 还是连不上：{e}")