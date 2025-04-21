import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

load_dotenv()

def get_response(question: str) -> str:
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)
    return qa.run(question)