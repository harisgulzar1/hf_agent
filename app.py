import os
from dotenv import load_dotenv
import requests
import pandas as pd
import inspect

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables from .env file
load_dotenv()

# Get values
HF_USERNAME = os.getenv("HF_USERNAME")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_TOKEN")
API_URL = os.getenv("API_URL", "https://jofthomas-unit4-scoring.hf.space/")  # Fallback to default


# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

search_tool = DuckDuckGoSearchRun()

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [search_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

# class BasicAgent:
#     def __call__(self, question: str) -> str:
#         return "This is a default answer."

def run_and_submit_all():
    if not HF_USERNAME:
        print("HF_USERNAME is not set in the environment.")
        return

    questions_url = f"{API_URL}/questions"
    submit_url = f"{API_URL}/submit"

    # agent = BasicAgent()
    agent_code = inspect.getsource(AgentState)

    try:
        response = requests.get(questions_url, timeout=10)
        response.raise_for_status()
        questions = response.json()
    except Exception as e:
        print("Error fetching questions:", e)
        return

    answers = []
    for q in questions:
        answer = alfred.invoke({"messages": q["question"]})
        answers.append({
            "task_id": q["task_id"],
            "submitted_answer": answer
        })

    submission = {
        "username": HF_USERNAME,
        "agent_code": agent_code,
        "answers": answers
    }

    try:
        res = requests.post(submit_url, json=submission, timeout=15)
        res.raise_for_status()
        result = res.json()
        print("Submission Result:", result)
    except Exception as e:
        print("Submission failed:", e)

if __name__ == "__main__":
    run_and_submit_all()
