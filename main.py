import asyncio
from agents.research_agent import ResearchAgent
from vectorstore.vector import VectorStore
from llm.answer_drafter import draft_answer

async def run(query: str):
    print(f"🔍 Searching for: {query}")
    researcher = ResearchAgent()
    chunks = await researcher.gather_information(query)
    print(f"✅ Retrieved {len(chunks)} cleaned chunks")

    vectorstore = VectorStore()
    vectorstore.build_index(chunks)

    top_chunks = vectorstore.search(query)
    print(f"📚 Selected {len(top_chunks)} most relevant chunks")

    answer = draft_answer(top_chunks, query)
    print(f"""\n💬 Final Answer:\n{answer}""")

if __name__ == "__main__":
    asyncio.run(run("What is the future of quantum computing?"))