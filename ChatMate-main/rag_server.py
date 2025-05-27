import re
from flask import Flask, request, Response
from flask_cors import CORS
import json
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
import rag_test.config as cfg

from langchain.chains.base import Chain
from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
CORS(app)

role = {
    'bot': 'assistant',
    'user': 'user'
}

class CleanRetrievalQA(Chain):
    """自定义的RetrievalQA链，确保干净的输出"""
    
    retriever: Any
    llm: Any
    
    @property
    def input_keys(self) -> List[str]:
        return ["query"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["result"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        chat_history = inputs.get("chat_history", [])
        
        # 1. 检索相关文档
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. 使用自定义提示模板生成回答
        prompt_template = ChatPromptTemplate.from_template("""
        你是一位专业医生助理，请根据以下医学知识回答问题。
        知识内容：{context}
        对话历史：{history}
        问题：{question}
        要求：
        - 直接回答问题，禁止重复问题或上下文，不要思考过程
        - 禁止自我重复
        - 回答要简明扼要，分点说明
        - 不要出现"根据资料"、"上下文显示"等提示词
        - 如果不知道答案，如实告知
        """)
        
        chain = prompt_template | self.llm | StrOutputParser()
        result = ''
        while result == '':
            result = chain.invoke({"context": context, "question": query, "history":chat_history})
            # 3. 清理输出
            while result!='':
                result = result.strip()
                if result.startswith("Assistant:"):
                    result = result[10:]
                elif result.startswith("Assistant :"):
                    result = result[11:]
                else:
                    break
                print("piece:"+result)
            result = re.sub(r"（注：.*", "", result, flags=re.DOTALL).strip()
            result = re.sub(r"Assistant.*", "", result, flags=re.DOTALL).strip()
            if result.startswith("好的，"):
                result = result[3:]
            if result.startswith("以下是"):
                result = result[3:]
            print("ans:"+result)
            
        return {"result": result}

# 在init_rag_components中使用自定义链

    


# 初始化RAG相关组件
def init_rag_components(base_url, api_key, model):
    # 根据配置选择嵌入模型
    if cfg.EMBEDDINGS_BACKEND == cfg.openai.OPENAI:
        embeddings = OpenAIEmbeddings(
            model=cfg.openai.EMBEDDINGS_MODEL,
            api_key=cfg.openai.OPENAI_API_KEY,
            base_url=cfg.openai.OPENAI_URL_BASE,
        )
    else:
        raise ValueError(f"Unsupported embeddings backend: {cfg.EMBEDDINGS_BACKEND}")
    
    # 根据配置选择LLM
    if cfg.LLM_BACKEND == cfg.openai.OPENAI:
        llm = OpenAI(
            model=cfg.openai.MODEL,
            temperature=cfg.LLM_TEMPERATURE,
            api_key=cfg.openai.OPENAI_API_KEY,
            base_url=cfg.openai.OPENAI_URL_BASE,
        )
    elif cfg.LLM_BACKEND == cfg.openai2.OPENAI:
        llm = OpenAI(
            model=model if model else cfg.openai2.MODEL,
            temperature=cfg.LLM_TEMPERATURE,
            api_key=api_key if api_key else cfg.openai2.OPENAI_API_KEY,
            base_url=base_url if base_url else cfg.openai2.OPENAI_URL_BASE,
        )
    else:
        raise ValueError(f"Unsupported LLM backend: {cfg.LLM_BACKEND}")
    
    # 初始化向量数据库
    if cfg.rag_db.USING_CACHE_DB and os.path.exists(cfg.rag_db.DB_PATH):
        vector_store = Chroma(
            persist_directory=cfg.rag_db.DB_PATH,
            embedding_function=embeddings,
        )
    else:
        shutil.rmtree(cfg.rag_db.DB_PATH) if os.path.exists(cfg.rag_db.DB_PATH) else None
        os.makedirs(cfg.rag_db.DB_PATH, exist_ok=True)
        vector_store = build_rag_db(embeddings)
    
    print(f"知识库加载完成，总条目数: {vector_store._collection.count()}")
    
    # 创建问答链
    qa_chain = CleanRetrievalQA(
        retriever=vector_store.as_retriever(),
        llm=llm
    )
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=vector_store.as_retriever(),
    #     chain_type="stuff"
    # )
    
    return qa_chain

def build_rag_db(embeddings):
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=cfg.rag_db.DB_PATH,
    )
    
    for filepath in cfg.rag_db.FILE_PATHS:
        try:
            # 确保文件存在
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"文档文件不存在: {filepath}")
            
            loader = PyPDFLoader(filepath) if filepath.endswith('.pdf') else TextLoader(filepath)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=cfg.rag_db.SEPERATORS,
                chunk_size=cfg.rag_db.CHUNK_SIZE,
                chunk_overlap=cfg.rag_db.CHUNK_OVERLAP_SIZE,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # 分批添加文档，避免一次性添加过多
            batch_size = 20
            for i in range(0, len(chunks), batch_size):
                vector_store.add_documents(documents=chunks[i:i+batch_size])
            
            print(f"文档已处理并添加到向量数据库: {filepath}")
        except Exception as e:
            print(f"处理文档 {filepath} 时出错: {str(e)}")
            continue
    
    return vector_store

def generate_stream_response_by_openai(messages=None, model='gpt-4o-mini', base_url=None, api_key=None):
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    for chunk in response:
        answer_chunk = chunk.choices[0].delta.content
        if answer_chunk not in ('', None):
            yield f"data: {json.dumps({'text': answer_chunk, 'reason': False}, ensure_ascii=False)}\n"
        yield ""

def generate_stream_response_by_openai2(messages=None, model='gpt-4o-mini', base_url=None, api_key=None, use_rag=True):
    if use_rag:
        try:
            # 初始化RAG组件
            qa_chain = init_rag_components(base_url, api_key, model)
            
            print(messages)
            # 获取最后一个用户消息
            last_user_message = next(
                (msg['content'] for msg in reversed(messages) if msg['role'] == 'user'),
                ""
            )
            processed_messages = messages
            
            if not last_user_message:
                yield f"data: {json.dumps({'text': '未获取到用户问题', 'reason': True}, ensure_ascii=False)}\n"
                yield ""
                return
            
            # 使用RAG获取答案
            result = qa_chain({'query': last_user_message, 'chat_history': processed_messages})
            print(result)
            rag_answer = result['result']
            
            # 模拟流式响应
            chunk_size = 10
            for i in range(0, len(rag_answer), chunk_size):
                yield f"data: {json.dumps({'text': rag_answer[i:i+chunk_size], 'reason': False}, ensure_ascii=False)}\n"
            yield ""
        except Exception as e:
            yield f"data: {json.dumps({'text': f'RAG处理出错: {str(e)}', 'reason': True}, ensure_ascii=False)}\n"
            yield ""
    else:
        # 原始OpenAI流式响应
        from openai import OpenAI
        
        client = OpenAI(
            base_url=base_url or cfg.openai2.OPENAI_URL_BASE,
            api_key=api_key or cfg.openai2.OPENAI_API_KEY,
        )

        response = client.chat.completions.create(
            model=model or cfg.openai2.MODEL,
            messages=messages,
            stream=True
        )

        for chunk in response:
            answer_chunk = chunk.choices[0].delta.content
            if answer_chunk not in ('', None):
                yield f"data: {json.dumps({'text': answer_chunk, 'reason': False}, ensure_ascii=False)}\n"
            yield ""

@app.route('/stream_openai_generate', methods=['POST'])
def stream_generate_openai():
    data = request.json
    messages = data['messages']
    model = data.get('model', '')
    base_url = data.get('base_url', '')
    api_key = data.get('api_key', '')
    use_rag = data.get('use_rag', False)

    # 构建消息历史
    processed_messages = []
    for line in messages[1:]:  # 跳过系统消息
        processed_messages.append({
            'role': role[line['sender']],
            'content': line['text']
        })

    if not base_url:
        return Response(
            generate_stream_response_by_openai(
                messages=processed_messages,
                model=model,
                api_key=api_key
            ),
            mimetype='text/event-stream',
            headers={
                'X-Accel-Buffering': 'no',
                'Cache-Control': 'no-cache'
            }
        )
    else:
        print("rag success")
        return Response(
            generate_stream_response_by_openai2(
                messages=processed_messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                # use_rag=use_rag
            ),
            mimetype='text/event-stream',
            headers={
                'X-Accel-Buffering': 'no',
                'Cache-Control': 'no-cache'
            }
        )

if __name__ == '__main__':
    # 检查并创建必要的目录
    os.makedirs(cfg.rag_db.DB_PATH, exist_ok=True)
    for filepath in cfg.rag_db.FILE_PATHS:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)