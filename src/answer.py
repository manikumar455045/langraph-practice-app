from dotenv import load_dotenv
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mani-resume-finder"

DB_NAME = str(Path(__file__).parent.parent / "db" / "mani_resume_db")
vector_store=Chroma(persist_directory=DB_NAME, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
retriever = vector_store.as_retriever()

system_prompt = """You are a helpful AI assistant who is reviewing the resume of a job candidate and answers questions based on the information provided in the context below
Use the given context only to answer the question and be descriptive about the role.
If you don't know the answer, say you don't know.
context: {context}"""

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.7)


def answer_question(question):
    try:
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        formatted_prompt = system_prompt.format(context=context)
        messages = [SystemMessage(content=formatted_prompt), HumanMessage(content=question)]
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    question = "What are Mani's skills, years of experience and the companies he has worked for?, and which is he currently working for?"
    # question = "Did Mani lead a team in Deloitte and is he currently employed there?"
    # question = "What is Mani's educational background?"
    # question = "Was mani able to improve any coversion rates for the companies he worked for and if so by how much?"

    answer = answer_question(question)
    print(f"Question: {question}\nAnswer: {answer}")