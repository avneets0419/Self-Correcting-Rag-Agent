from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> str:
    results = client.search(query, max_results=3)
    return "\n\n".join([r["content"] for r in results["results"]])