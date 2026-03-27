from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
import json
import os

# ── KNOWLEDGE GAP DETECTOR
class KnowledgeGapDetector:
    def __init__(self, log_file="data/knowledge_gaps.json"):
        self.log_file = log_file
        self.gaps = self.load_gaps()

    def load_gaps(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                return json.load(f)
        return []

    def save_gaps(self):
        with open(self.log_file, "w") as f:
            json.dump(self.gaps, f, indent=2)

    def log_gap(self, query, confidence):
        if confidence < 0.5:
            gap = {
                "query": query,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "status": "UNANSWERED"
            }
            self.gaps.append(gap)
            self.save_gaps()
            print(f"\n⚠️  Knowledge Gap Detected!")
            print(f"   Query: '{query}'")
            print(f"   Confidence: {confidence:.0%}")
            print(f"   Logged to: {self.log_file}")


# ── MAIN RAG CLASS
class KnowledgeCopilot:
    def __init__(self):
        print("🔄 Knowledge Copilot load ho raha hai...")

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Vector DB load karo
        self.vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        # LLM
        self.llm = Ollama(model="llama3.1")

        # Retriever — Top 3 documents
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={"k": 3}
        )

        # Knowledge Gap Detector
        self.gap_detector = KnowledgeGapDetector()

        # Prompt
        self.prompt = PromptTemplate.from_template("""
Tum ek helpful Enterprise Knowledge Copilot ho.
Neeche diye gaye context ke basis pe question ka jawab do.
Agar context mein answer nahi hai toh honestly bolo:
"Mujhe is topic pe documentation nahi mili."

Context:
{context}

Question: {question}

Answer (Hindi ya English mein):""")

        # RAG Chain
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print("✅ Knowledge Copilot ready!\n")

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _calculate_confidence(self, query):
        """Confidence score calculate karo"""
        docs = self.vectordb.similarity_search_with_score(query, k=3)
        if not docs:
            return 0.0
        # Score ko 0-1 mein convert karo
        avg_score = sum(score for _, score in docs) / len(docs)
        # ChromaDB mein lower score = better match
        confidence = max(0, 1 - (avg_score / 2))
        return round(confidence, 2)

    def ask(self, question):
        """Question poochho aur answer lo"""
        print("\n" + "─"*50)

        # Confidence check karo
        confidence = self._calculate_confidence(question)

        # Sources dhundho
        sources = self.retriever.invoke(question)

        # Answer generate karo
        print("🤔 Soch raha hun...")
        answer = self.chain.invoke(question)

        # Knowledge gap log karo
        self.gap_detector.log_gap(question, confidence)

        # Result return karo
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources
        }

    def display_result(self, result):
        """Result display karo"""
        # Confidence color
        conf = result["confidence"]
        if conf >= 0.7:
            conf_emoji = "🟢"
        elif conf >= 0.4:
            conf_emoji = "🟡"
        else:
            conf_emoji = "🔴"

        print(f"\n✅ ANSWER:")
        print(f"{result['answer']}")

        print(f"\n{conf_emoji} Confidence: {conf:.0%}")

        print(f"\n📄 SOURCES:")
        for i, doc in enumerate(result["sources"]):
            source = doc.metadata.get("source", "Unknown")
            source = os.path.basename(source)
            print(f"  {i+1}. [{source}]")
            print(f"     {doc.page_content[:100]}...")

        print("─"*50)


# ── MAIN LOOP
if __name__ == "__main__":
    print("="*50)
    print("🚀 ENTERPRISE KNOWLEDGE COPILOT")
    print("="*50)

    copilot = KnowledgeCopilot()

    print("💡 Sample Questions:")
    print("  - Password reset kaise karte hain?")
    print("  - VPN setup kaise karo?")
    print("  - Leave apply kaise karte hain?")
    print("  - P0 incident mein kya karna chahiye?")
    print("\nExit karne ke liye: 'bye' likho\n")

    while True:
        question = input("❓ Tumhara sawaal: ").strip()

        if not question:
            continue

        if question.lower() in ["bye", "exit", "quit"]:
            print("\n👋 Bye! All the best!")
            break

        result = copilot.ask(question)
        copilot.display_result(result)