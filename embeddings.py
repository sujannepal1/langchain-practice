from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# TODO make this dynamic
# embed the txt as we add new data


class Embedding:
    def text_loader(self):
        loader = TextLoader("./merchanttrade.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()

        return docs, embeddings
