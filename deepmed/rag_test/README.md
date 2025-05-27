# env

```sh
# 环境安装（基于python3.11）
pip install langchain-community unstructured pymupdf langchain-text-splitters tiktoken langchain-chroma sentence-transformers
pip install langchain-openai langchain-ollama
pip install langchain-openai -i https://pypi.python.org/simple/ # 似乎镜像源无法安装langchain-openai，需要用该命令指定官方源
```
具体配置和实现原理介绍见`main.py`, `config.py`中的注释文字
**注意**：记得修改`config.py`中的`rag_db.USING_CACHE_DB`，如果为`False`，则会删除已有数据库，重新由文档生成数据库，很耗时；如果为`True`，则会使用已有数据库，不会重新生成

# 参考资料

- [RAG入门实践：手把手Python实现搭建本地知识问答系统](https://blog.csdn.net/arbboter/article/details/145758644)
	- 代码基于该部分实现
- [RAG文本分块策略](https://zhuanlan.zhihu.com/p/38753080790)
	- 最初使用默认的分块参数，发现效果不好。因此看到这篇文章
	- 而且由于这个文档信息密度较高，因此默认分段长度太长，这里也缩短了
	- [理解LangChain的RecursiveCharacterTextSplitter](https://zhuanlan.zhihu.com/p/650876562)
	- [深入解析 RecursiveCharacterTextSplitter 类 langchain_text_splitters.charater.py](https://blog.csdn.net/xycxycooo/article/details/141894742)