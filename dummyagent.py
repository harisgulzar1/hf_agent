import os
from dotenv import load_dotenv
import requests
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers import LangChainTracer
from langsmith import traceable
from langgraph.types import Command


# Load .env
load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = os.getenv("API_URL")

# Dummy Tool
@tool
def dummy_tool(placeholder: str) -> str:
    '''
    Dummy tool called from the agent in the tools node.
    '''
    print("✅ DUMMY TOOL EXECUTED")
    return f"Echo: {placeholder}"

# Tools + ToolNode
tools = [dummy_tool]
tool_node = ToolNode(tools=tools)

# LLM
llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    callbacks=[LangChainTracer()],
).bind_tools(tools)

# Agent State
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration: int


def should_continue(state: AgentState):
    messages = state["messages"]
    iteration = state["iteration"]

    last_message = messages[-1]
    if last_message.tool_calls and iteration < 2:
        return "tools"
    return END

# Main logic
@traceable(name="main")
def main_node(state: AgentState) -> AgentState:
    iteration = state["iteration"]
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response],
            "iteration": iteration + 1}


# LangGraph
builder = StateGraph(AgentState)
builder.add_node("main", main_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("main")
builder.add_conditional_edges("main", should_continue, ["tools", END])
builder.add_edge("tools", "main")
builder.set_finish_point("main")

alfred = builder.compile()

with open("dummy_agent.png", "wb") as f:
    f.write(alfred.get_graph().draw_mermaid_png())

# HF runner
@traceable(name="run_minimal_tool_agent")
def run_and_submit_all():
    if not HF_USERNAME:
        print("❌ HF_USERNAME is not set.")
        return

    questions_url = f"{API_URL}/questions"
    submit_url = f"{API_URL}/submit"

    try:
        response = requests.get(questions_url, timeout=10)
        response.raise_for_status()
        questions = response.json()
    except Exception as e:
        print("Error fetching questions:", e)
        return

    answers = []
    for q in questions:
        input_content = q.get("question", "")
        task_id = q.get("task_id")

        print(f"\n▶ Question: {input_content}")
        state = {
            "messages": [HumanMessage(content=input_content)],
            "iteration": 0
        }
        result = alfred.invoke(state)
        answer = result["messages"][-1].content
        print("✅ Answer:", answer)

        answers.append({
            "task_id": task_id,
            "submitted_answer": answer,
        })

    submission = {
        "username": HF_USERNAME,
        "answers": answers
    }

    try:
        res = requests.post(submit_url, json=submission, timeout=15)
        res.raise_for_status()
        print("🎉 Submission Result:", res.json())
    except Exception as e:
        print("❌ Submission failed:", e)

if __name__ == "__main__":
    run_and_submit_all()
