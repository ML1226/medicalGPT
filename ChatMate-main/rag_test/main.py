# main.py

# 0. 安装环境
# pip install langchain-community unstructured pymupdf langchain-text-splitters tiktoken langchain-chroma sentence-transformers
# pip install langchain-openai langchain-ollama
# pip install langchain-openai -i https://pypi.python.org/simple/ # 似乎镜像源无法安装langchain-openai，需要用该命令指定官方源

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
import shutil
# from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import OpenAIEmbeddings, OpenAI
import config as cfg

# 1. 初始化大模型和嵌入模型
def init_llm_backend():
    if cfg.EMBEDDINGS_BACKEND == cfg.openai.OPENAI:
        embeddings = OpenAIEmbeddings(
            model=cfg.openai.EMBEDDINGS_MODEL,
            api_key=cfg.openai.OPENAI_API_KEY,
            base_url=cfg.openai.OPENAI_URL_BASE,
        )
    # elif cfg.EMBEDDINGS_BACKEND == cfg.ollama.OLLAMA:
    #     embeddings = OllamaEmbeddings(
    #         model=cfg.ollama.EMBEDDINGS_MODEL,
    #         base_url=cfg.ollama.OLLAMA_HOST,
    #     )
    else:
        raise ValueError(f"Unknown embeddings backend: {cfg.EMBEDDINGS_BACKEND}")
    
    if cfg.LLM_BACKEND == cfg.openai.OPENAI:
        llm = OpenAI(
            model=cfg.openai.MODEL,
            temperature=cfg.LLM_TEMPERATURE,
            api_key=cfg.openai.OPENAI_API_KEY,
            base_url=cfg.openai.OPENAI_URL_BASE,
        )
    elif cfg.LLM_BACKEND == cfg.openai2.OPENAI:
        llm = OpenAI(
            model=cfg.openai2.MODEL,
            temperature=cfg.LLM_TEMPERATURE,
            api_key=cfg.openai2.OPENAI_API_KEY,
            base_url=cfg.openai2.OPENAI_URL_BASE,
        )
    # elif cfg.LLM_BACKEND == cfg.ollama.OLLAMA:
    #     llm = OllamaLLM(
    #         model=cfg.ollama.MODEL,
    #         temperature=cfg.LLM_TEMPERATURE,
    #         base_url=cfg.ollama.OLLAMA_HOST,
    #     )
    else:
        raise ValueError(f"Unknown LLM backend: {cfg.LLM_BACKEND}")
    return embeddings, llm

# 2. 对文档数据进行处理，储存到向量数据库（若已储存，则跳过）
def build_rag_db(embeddings):
    # 2.1 初始化向量数据
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=cfg.rag_db.DB_PATH,
    )
    for idx, filepath in enumerate(cfg.rag_db.FILE_PATHS):
        # 2.2 文档加载
        loader = PyPDFLoader(filepath) if filepath.endswith('.pdf') else TextLoader(filepath)
        documents = loader.load()
        # 查看已加载的部分文档内容
        # print(documents[50].page_content[:100])
        # exit(0)

        # 2.3 文档分块
        text_splitter = RecursiveCharacterTextSplitter(
            separators=cfg.rag_db.SEPERATORS,
            chunk_size=cfg.rag_db.CHUNK_SIZE,
            chunk_overlap=cfg.rag_db.CHUNK_OVERLAP_SIZE,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        # 查看分块后的部分文档内容
        # print(len(chunks))
        # print(chunks[800].page_content)
        # exit(0)
        # 2.4 使用嵌入模型将分块文档计算成向量数据
        for i in range(len(chunks)//20+1):
            # print(i)
            vector_store.add_documents(documents=chunks[i*20:(i+1)*20]) # 似乎不允许一次性添加太多，目前发现50条/次会报错，20条/次则不会
        print(f'文档已追加: {filepath}')
    # 2.5 保存到向量数据库（这个功能是根据persist_directory参数自动执行的，不需要额外代码）
    return vector_store

# 3. 加载向量数据库
def init_rag_db(embeddings):
    if cfg.rag_db.USING_CACHE_DB:
        vector_store = Chroma(
            persist_directory=cfg.rag_db.DB_PATH,
            embedding_function=embeddings,
        )
    else:
        shutil.rmtree(cfg.rag_db.DB_PATH) if os.path.exists(cfg.rag_db.DB_PATH) else None
        os.mkdir(cfg.rag_db.DB_PATH)
        vector_store = build_rag_db(embeddings)
    print(f"成功加载知识库，条目数：{vector_store._collection.count()}")
    return vector_store

# 5. 测试问答 (在此处修改问答内容)
def test(qa_chain):
    query = '我家老人最近比较担心心脑血栓相关的疾病，有什么相关介绍吗？'
    
    print('====QUERY====')
    print(query)
    
    result = qa_chain({'query': query})
    
    print('====RESULT====')
    print(result['result'])

if __name__ == '__main__':
    embeddings, llm = init_llm_backend()
    vector_store = init_rag_db(embeddings=embeddings)
    
    # 4. 构造问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff"
    )
    
    test(qa_chain)
    