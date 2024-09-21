import os
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate


class Llm:
    @classmethod
    def get_mixtral_llm(cls):
        # Define the repo ID and connect to Mixtral model on Huggingface
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        )
        return llm

    @classmethod
    def get_prompt(cls):
        template = """
        You are a chatbot. These Human will ask you a questions about the app or the process they are confused in. 
        Use following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer: 

        """

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        return prompt
