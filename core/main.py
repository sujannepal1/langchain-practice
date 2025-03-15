from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from embedding_model.embeddings import Embedding
from llm.llm import Llm
from vector_db.vectorized_db import VectorizedDB


class ChatBot:
    embeddings = Embedding.text_loader()
    docs, embeds = embeddings
    doc_search = VectorizedDB().pinecone_init(docs, embeds)
    llm = Llm.get_mixtral_llm()
    prompt = Llm.get_prompt()

    rag_chain = (
        {"context": doc_search.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    bot = ChatBot()
    input = input("Ask me anything: ")
    result = bot.rag_chain.invoke(input)
    print(result)
