from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os

loader = PyPDFLoader("/workspaces/Reading_docs_GEMINI/Reposición_ESIM_Manual.pdf")
data = loader.load()
print(data)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs=text_splitter.split_documents(data)
len(docs)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'f5444e56-58db-42db-afd6-d4bd9b2cb40c')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'asia-southeast1-gcp-free')

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchainpinecone" # put in the name of your pinecone index here
docsearch=Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

#query="What are examples of good data science teams?"
query="YOLOv7 outperforms which models"
docs=docsearch.similarity_search(query)

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Loading model,
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=1024,
    verbose=False,
)

chain=load_qa_chain(llm, chain_type="stuff")
llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
chain=load_qa_chain(llm, chain_type="stuff")
query="What are examples of good data science teams?"
docs=docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)


