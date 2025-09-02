import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embedding

from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()





PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdf_file("C:/Users/manib/OneDrive/Desktop/project 9/luna-medical-adviser/data")
minimal_docs  = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)
embedding = download_hugging_face_embedding()


pinecone_api_key = PINECONE_API_KEY

PC = Pinecone(api_key=pinecone_api_key)


#from pinecone import ServerlessSpec

index_name = "lunamedicalbot"

if not PC.has_index(index_name):
    
    PC.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        
    )
index = PC.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)