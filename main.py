from embeddings import Embedding
from llm import Llm
from verctoriz_db import VectorizedDB


embeddings = Embedding().text_loader()
docs, embeds = embeddings
docsearch = VectorizedDB().pinecone_init(docs, embeds)
llm = Llm.get_mixtral_llm()
prompt = Llm.get_prompt()

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

rag_chain = (
    {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
