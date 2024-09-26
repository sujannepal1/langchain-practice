import os

import pinecone
from langchain_community.vectorstores import Pinecone


class VectorizedDB:
    def __init__(self):
        # get from env
        self.index_name = "langchain-demo-384-dim"

    def pinecone_init(self, docs, embeddings):
        # Initialize Pinecone client
        # TODO make config
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
        if self.index_name not in pinecone.list_indexes():
            # Create new Index
            pinecone.create_index(name=self.index_name, metric="cosine", dimension=384)
            doc_search = Pinecone.from_documents(
                docs, embeddings, index_name=self.index_name
            )
        else:
            # Link to the existing index
            doc_search = Pinecone.from_existing_index(self.index_name, embeddings)

        return doc_search
    
    def qdrant(self):
        # TODO make configs for qdrant:
        # https://qdrant.tech/documentation/quickstart/
        pass
        
