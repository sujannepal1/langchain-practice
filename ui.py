from dotenv import load_dotenv
from langchain.document_loaders import TextLoader


class ChatBot:
    load_dotenv()
    loader = TextLoader("./merchanttrade.txt")
    documents = loader.load()

    # The rest of the code here

    rag_chain = (
        {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
