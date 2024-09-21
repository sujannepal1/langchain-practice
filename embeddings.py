from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# TODO make this dynamic
# embed the txt as we add new data

FILE_NAME = "./merchanttrade.txt"
HOROSCOPE_FILE_PATH = "./horoscope.txt"


class Embedding:
    @classmethod
    def text_loader(cls):
        loader = TextLoader(HOROSCOPE_FILE_PATH)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return docs, embeddings
