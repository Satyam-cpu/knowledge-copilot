import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import json
from datetime import datetime

# Environment variables load karo
load_dotenv()

# ── TICKET LOOKUP SYSTEM
class TicketLookup:
    def __init__(self, csv_path="data/tickets/tickets.csv"):
        # Folder check taaki error na aaye
        if not os.path.exists(csv_path):
             os.makedirs(os.path.dirname(csv_path), exist_ok=True)
             # Dummy empty dataframe agar file nahi hai
             self.df = pd.DataFrame(columns=['id', 'title', 'description', 'category', 'resolution', 'status'])
        else:
            self.df = pd.read_csv(csv_path)
        print(f"✅ Tickets loaded: {len(self.df)} tickets")

    def search(self, keywords: str) -> str:
        """Keywords se similar tickets dhundho"""
        if self.df.empty: return "Database mein koi tickets nahi hain."
        
        keywords_list = keywords.lower().split()
        results = []

        for _, row in self.df.iterrows():
            text = f"{row['title']} {row['description']} {row['category']}".lower()
            matches = sum(1 for kw in keywords_list if kw in text)
            if matches > 0:
                results.append((matches, row))

        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:3]

        if not top_results:
            return "Koi similar ticket nahi mila past records mein."

        output = f"Top {len(top_results)} similar past tickets mile:\n\n"
        for i, (score, row) in enumerate(top_results):
            output += f"Ticket {i+1}:\n"
            output += f"  ID: {row['id']}\n"
            output += f"  Title: {row['title']}\n"
            output += f"  Category: {row['category']}\n"
            output += f"  Resolution: {row['resolution']}\n"
            output += f"  Status: {row['status']}\n\n"

        return output


# ── DOCUMENT SUMMARIZER
class DocumentSummarizer:
    def __init__(self, llm, vectordb):
        self.llm = llm
        self.vectordb = vectordb

    def summarize(self, topic: str) -> str:
        """Topic ke baare mein documents ka summary do"""
        docs = self.vectordb.similarity_search(topic, k=4)

        if not docs:
            return f"'{topic}' ke baare mein koi document nahi mila."

        combined = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Neeche diye gaye documents ko padho aur '{topic}' ke baare mein
ek clear aur concise summary do — bullet points mein.

Documents:
{combined}

Summary:"""

        response = self.llm.invoke(prompt)
        return response.content


# ── KNOWLEDGE GAP LOGGER
class GapLogger:
    def __init__(self, log_file="data/knowledge_gaps.json"):
        self.log_file = log_file
        os.makedirs("data", exist_ok=True)

    def log(self, query: str, reason: str = "Low confidence"):
        gaps = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    gaps = json.load(f)
            except: gaps = []

        gaps.append({
            "query": query,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

        with open(self.log_file, "w") as f:
            json.dump(gaps, f, indent=2)

        return f"⚠️ Knowledge Gap logged: '{query}'"


# ── MAIN AGENT CLASS
class KnowledgeAgent:
    def __init__(self):
        print("🔄 Knowledge Agent load ho raha hai...")

        # Groq LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        # Embeddings + Vector DB
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        # Sub-systems
        self.ticket_lookup = TicketLookup()
        self.summarizer = DocumentSummarizer(self.llm, self.vectordb)
        self.gap_logger = GapLogger()

        # Tools banao
        self.tools = self._create_tools()

        # Agent banao
        self.agent_executor = self._create_agent()

        print("✅ Knowledge Agent ready!\n")

    def _create_tools(self):
        """3 Agentic Tools banao"""

        def document_search(query: str) -> str:
            docs = self.vectordb.similarity_search(query, k=3)
            if not docs:
                return "Koi relevant document nahi mila."
            result = ""
            for i, doc in enumerate(docs):
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                result += f"[Source: {source}]\n{doc.page_content}\n\n"
            return result

        def ticket_lookup_func(keywords: str) -> str:
            return self.ticket_lookup.search(keywords)

        def summarize_func(topic: str) -> str:
            return self.summarizer.summarize(topic)

        return [
            Tool(
                name="document_search",
                description="Use for searching company policies, SOPs, or internal guides. Input: search query.",
                func=document_search
            ),
            Tool(
                name="ticket_lookup",
                description="Search past support tickets for similar issues and resolutions. Input: keywords.",
                func=ticket_lookup_func
            ),
            Tool(
                name="summarizer",
                description="Provides a detailed summary of a specific topic from documents. Input: topic.",
                func=summarize_func
            ),
        ]

    def _create_agent(self):
        """ReAct Agent banao"""
        
        # FIXED: ReAct prompt must include {agent_scratchpad}, {tools}, and {tool_names}
        template = """Tum ek helpful Enterprise Knowledge Copilot ho.
Tumhare paas ye tools hain:
{tools}

In tools ka naam use karo: {tool_names}

Hamesha niche diye gaye format ka use karo:

Question: input question
Thought: Tumhe kya karna chahiye?
Action: tool ka naam [{tool_names}] mein se ek
Action Input: tool ka input
Observation: tool ka result
... (ye steps repeat ho sakte hain)
Thought: Ab mujhe final answer pata hai
Final Answer: Aapka complete response with sources if any

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)

        # FIXED: Calling directly without 'langchain.agents.' prefix
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # FIXED: Using imported AgentExecutor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def ask(self, question: str) -> str:
        """Question poochho"""
        try:
            result = self.agent_executor.invoke({"input": question})
            return result["output"]
        except Exception as e:
            self.gap_logger.log(question, str(e))
            return f"Error: {str(e)}\nKnowledge gap log ho gayi hai."


# ── MAIN
if __name__ == "__main__":
    print("="*50)
    print("🤖 KNOWLEDGE AGENT — ReAct Mode")
    print("="*50)

    agent = KnowledgeAgent()

    print("💡 Sample Questions:")
    print("  - VPN issue ho raha hai — koi past ticket tha?")
    print("  - Password policy ka summary do")
    print("\nExit: 'bye'\n")

    while True:
        question = input("❓ Sawaal: ").strip()
        if not question: continue
        if question.lower() in ["bye", "exit", "quit"]: break

        print("\n🤔 Agent soch raha hai...\n")
        answer = agent.ask(question)

        print("\n" + "="*50)
        print("✅ FINAL ANSWER:")
        print(answer)
        print("="*50 + "\n")