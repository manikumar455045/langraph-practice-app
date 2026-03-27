from dotenv import load_dotenv
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
load_dotenv()
DB_NAME = str(Path(__file__).parent.parent / "db" / "mani_resume_db")
vector_store=Chroma(persist_directory=DB_NAME, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
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
        print(f"Context:\n{context}\n")
        messages = [SystemMessage(content=formatted_prompt), HumanMessage(content=question)]
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # question = "What are Mani's skills, years of experience and the companies he has worked for?"
    # question = "Did Mani lead a team in Deloitte and is he currently employed there?"
    question = "What is Mani's educational background?"

    answer = answer_question(question)
    print(f"Question: {question}\nAnswer: {answer}")