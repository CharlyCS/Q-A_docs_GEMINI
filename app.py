from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import sys

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("Q_A_docs_GEMINI/faiss_index")


def get_conversational_chain():
    prompt_template = """
    Responda la pregunta lo más detalladamente posible desde el contexto proporcionado, asegúrese de proporcionar todos los detalles, si la respuesta no está en el contexto proporcionado, simplemente diga "la respuesta no está disponible en el contexto", no proporcione la respuesta incorrecta\n\n
    Contexto:\n {context}?\n
    Pregunta: \n{question}\n

    Respuesta:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.8,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

    new_db = FAISS.load_local("Q_A_docs_GEMINI/faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    value = response.get('output_text')
    return print(value)

with open("Q_A_docs_GEMINI/docs/output.txt", "r", encoding="utf-8") as raw_text_file:
    raw_text = raw_text_file.read()

text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input(sys.argv[1])
    else:
        print("No input provided")