class rag_db:
    USING_CACHE_DB = True # NOTE：重要，如果为False，则会删除已有数据库，重新生成数据库，很耗时；如果为True，则会使用已有数据库，不会重新生成
    DB_PATH = '/root/autodl-tmp/MedicalGPT/MedicalGPT/ChatMate-main/rag_test/chroma_db'
    FILE_PATHS = ['assets/病理学 (卞修武,李一雷) (Z-Library).pdf', 'assets/诊断学 (万学红, 卢雪峰) (Z-Library).pdf']
    SEPERATORS = ['\n\n', '。', '！', '？', '；', '：', '，', '\n'] # 表示用于切分文本的字符/字符串，和区分语义的字符有关。由前到后尝试切分，直到长度不超过CHUNK_SIZE
    CHUNK_SIZE: int = 200 # 1024 # 表示每个chunk的最大长度，该值设定应和完整语义的长度有关
    CHUNK_OVERLAP_SIZE: int = 50 # 表示前一个chunk和后一个chunk重叠的长度，不知道怎么设置更好

class ollama:
    OLLAMA = 'ollama'
    OLLAMA_HOST = 'http://127.0.0.1:11434'
    EMBEDDINGS_MODEL = 'bge-m3:latest'
    MODEL = 'qwen2.5:1.5b'

class openai:
    OPENAI = 'openai'
    OPENAI_API_KEY = "sk-iuvkgsdmznhzpsylpsntvmrlwszpakenmuqyvmfucabmnepv"
    OPENAI_URL_BASE = "https://api.siliconflow.cn/v1"
    EMBEDDINGS_MODEL = 'BAAI/bge-m3'
    MODEL = 'Qwen/Qwen3-8B'

class openai2:
    OPENAI = 'openai2'
    OPENAI_API_KEY = 'fake_key'
    OPENAI_URL_BASE = 'http://localhost:8000/v1'
    MODEL = '/root/autodl-tmp/MedicalGPT/MedicalGPT/outputs-merged-outputs-sft-v1'

EMBEDDINGS_BACKEND = openai.OPENAI # 嵌入模型选择上方class openai中的设置
LLM_BACKEND = openai2.OPENAI # 大模型选择上方class openai中的设置
LLM_TEMPERATURE = 0.8