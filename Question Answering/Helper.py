import os
import pickle
import numpy as np
from pymongo import MongoClient
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import faiss
import base64
import sys
from langchain.chains.question_answering import load_qa_chain
import bson.binary
from bson.binary import Binary

# loading the docs using directory loader function of langchain
def load_docs(directory = './DATA'):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents
# documents = load_docs()

os.environ['OPENAI_API_KEY'] = "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"

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs
# docs = split_docs(documents)

def embedding():
    embeddings = OpenAIEmbeddings()
    query_result = embeddings.embed_query("Hello world")

    # Creating And Storing Vectors
    db = FAISS.from_documents(docs, embeddings)  # stores index of document vectors


    # create a MongoDB client
    client = MongoClient('mongodb://localhost:27017/')

    # get the database and collection
    db1 = client['2707']
    collection = db1['iserv']

    # store the FAISS index in MongoDB
    index_bytes = pickle.dumps(db)
    index_binary = Binary(index_bytes)
    collection.replace_one({}, {'iserv': index_binary}, upsert=True)

documents = load_docs()
docs = split_docs(documents)
embedding()


def Mongo():
    # create a MongoDB client
    client = MongoClient('mongodb://localhost:27017/')

    # get the database and collection
    db1 = client['2707']
    collection = db1['iserv']

    # get the sotred index from mongodb
    serialized_index = collection.find_one()['iserv']

    # encode the pickled data as base64
    index_bytes = base64.b64encode(serialized_index)

    # decode the base64 data before unpickling
    decoded_index_bytes = base64.b64decode(index_bytes)

    faiss_index = pickle.loads(decoded_index_bytes)
    
    #Defining Model
    model_name = "text-davinci-003"
    llm = OpenAI(model_name=model_name,
                temperature= 0,
                top_p= 1,
                n= 1
                )

    chain = load_qa_chain(llm, chain_type="stuff")
    return faiss_index, chain

faiss_index, chain = Mongo()

def get_similiar_docs(query, k=2, score=False):
    if score:
        similar_docs = faiss_index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = faiss_index.similarity_search(query, k=k)
    return similar_docs

# Function for Searching Similar Documents related to question based on Cosine Similarities score
def get_answer(query):
    similar_docs = get_similiar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer


# query = "question?"
# answer = get_answer(query)