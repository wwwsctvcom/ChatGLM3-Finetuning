from modelscope import snapshot_download

# https://www.modelscope.cn/models/ZhipuAI/chatglm3-6b/files
model_dir = snapshot_download("ZhipuAI/chatglm3-6b", cache_dir="./cache", revision = "v1.0.2")
    
