import os
from dotenv import load_dotenv

# ✅ Modern LangChain imports (modular style)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

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

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Simple retrieval chain without RetrievalQA
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain


def answer_faq(question: str, chat_history=None):
    """
    Retrieve and answer a FAQ question using LangChain's Conversational RAG.
    """
    chat_history = chat_history or []
    chain = make_faq_chain()

    answer = chain.invoke(question)

    return {
        "answer": answer,
        "sources": []
    }