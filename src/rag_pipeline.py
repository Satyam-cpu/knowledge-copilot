from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("Step 1: Document load ho raha hai...")
loader = TextLoader("data/company_policy.txt", encoding="utf-8")
documents = loader.load()
print(f"✅ Document loaded! Pages: {len(documents)}")

print("\nStep 2: Chunks ban rahe hain...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)
print(f"✅ Total chunks: {len(chunks)}")

print("\nStep 3: Embeddings ban rahi hain...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("✅ Embedding model ready!")

print("\nStep 4: Vector DB mein store ho raha hai...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("✅ Vector DB ready!")

print("\nStep 5: LLM connect ho raha hai...")
llm = Ollama(model="llama3.1")
print("✅ LLM ready!")

print("\nStep 6: RAG Chain ban rahi hai...")

# Prompt template
prompt = PromptTemplate.from_template("""
Tum ek helpful IT support assistant ho.
Neeche diye gaye context ke basis pe question ka jawab do.
Agar context mein answer nahi hai toh honestly bolo.

Context:
{context}

Question: {question}

Answer:
""")

# Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain — naya tarika
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG Chain ready!")
print("\n" + "="*50)
print("🚀 Knowledge Copilot Ready! Poochho kuch bhi!")
print("="*50)

# Questions loop
while True:
    question = input("\n❓ Tumhara sawaal: ")
    
    if question.lower() in ["exit", "quit", "bye"]:
        print("👋 Bye!")
        break
    
    print("\n🔍 Documents search ho rahe hain...")
    answer = rag_chain.invoke(question)
    
    print("\n✅ Answer:")
    print(answer)
    
    # Sources dikhao
    print("\n📄 Sources:")
    docs = retriever.invoke(question)
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc.page_content[:100]}...")