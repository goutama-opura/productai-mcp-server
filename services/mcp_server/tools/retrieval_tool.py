import os
from dotenv import load_dotenv

# ✅ Modern LangChain imports (modular style)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_PATH = os.getenv("FAQ_INDEX_PATH", "faiss_faq_index")

def make_faq_chain():
    """
    Build the RAG chain for answering FAQ-style questions.
    Loads FAISS index and creates a conversational retriever.
    """
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY is missing. Please check your .env file.")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True
    )
    return chain


def answer_faq(question: str, chat_history=None):
    """
    Retrieve and answer a FAQ question using LangChain's Conversational RAG.
    """
    chat_history = chat_history or []
    chain = make_faq_chain()

    result = chain.invoke({"question": question, "chat_history": chat_history})
    answer = result["answer"]
    sources = [
        d.page_content[:250] for d in result.get("source_documents", [])
    ]

    return {
        "answer": answer,
        "sources": sources
    }