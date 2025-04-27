# house_tour_rag_local.py

# 1. Install required packages FIRST
# pip install langchain openai faiss-cpu sentence-transformers tiktoken langchain-community

import os
from langchain.document_loaders import DirectoryLoader, TextLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # ✅ Correct import now

from dotenv import load_dotenv

# 2. Load environment variables
load_dotenv()

# 3. Set your OpenAI API Key (proper method)
openai_api_key = os.getenv("OPENAI_API_KEY") or "your-openai-api-key-here"

# 4. Folder path setup (your local folder)
folders = ["/Users/truptaditya/Documents/GitHub/HouseTour/Images"]  # 📂 Adjust this path as needed

# 5. Load documents
text_loader_kwargs = {'encoding': 'utf-8'}
documents = []

for folder in folders:
    doc_type = os.path.basename(folder.rstrip("/"))
    loader = DirectoryLoader(folder, glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"✅ Loaded {len(documents)} documents.")

# 6. Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(documents, embeddings)

# 7. Setup retriever and LLM
retriever = db.as_retriever()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",  # or "gpt-4o" if you prefer
    openai_api_key=openai_api_key,
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 8. Ask your queries
query1 = "Tell me about Kitchen"
answer1 = qa_chain.invoke(query1)
print("\n🛋️ Kitchen Info:\n", answer1['result'])

query2 = "Give me house tour"
answer2 = qa_chain.invoke(query2)
print("\n🏠 House Tour:\n", answer2['result'])

#Voice input function (listen_to_microphone)


