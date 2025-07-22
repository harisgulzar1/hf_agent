import os
from dotenv import load_dotenv
import requests
import pandas as pd
import inspect
from IPython.display import Image, display
import mimetypes
import base64
import tempfile
from langchain_core.tools import tool

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get values
HF_USERNAME = os.getenv("HF_USERNAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = os.getenv("API_URL", "https://jofthomas-unit4-scoring.hf.space/")

# OpenAI native client for tool use
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Web Search Tool using OpenAI's web_search_preview
@tool
def openai_web_search(query: str) -> str:
    """
    Perform a web search to find recent and relevant information from the internet.
    Takes a search query string and returns the summarized result.
    """
    print("WEB TOOL EXECUTED")
    response = openai_client.responses.create(
        model="gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input=query
    )
    return response.output_text

@tool
def image_analyzer_tool(image_path: str, prompt: str = "Describe this image.") -> str:
    """
    Analyze an image file and return a description using OpenAI's vision model.
    Takes the local image file path and an optional natural language prompt.
    """
    print("IMAGE TOOL EXECUTED")
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    base64_img = base64.b64encode(image_bytes).decode("utf-8")

    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who analyzes images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]
            }
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

@tool
def audio_transcription_tool(audio_path: str, prompt: str = "Transcribe this audio.") -> str:
    """
    Transcribe an audio file into text using OpenAI's Whisper model.
    Takes a local audio file path and returns the transcription.
    """
    print("AUDIO TOOL EXECUTED")
    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            prompt=prompt
        )
    return transcript.strip()

tools = [openai_web_search, image_analyzer_tool, audio_transcription_tool]
tool_node = ToolNode(tools)

llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    verbose=False
)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    input_file: str

def extract_final_answer(content: str) -> str:
    if "FINAL ANSWER:" in content:
        return content.split("FINAL ANSWER:", 1)[1].strip()
    return content.strip()

def assistant(state: AgentState):
    tool_descriptions = '''
openai_web_search(query: str) -> str:
    Perform a web search to find recent and relevant information.

image_analyzer_tool(image_path: str, prompt: str) -> str:
    Analyze the contents of an image file (such as input_file) using OpenAI's vision model.

audio_transcription_tool(audio_path: str, prompt: str) -> str:
    Transcribe an audio file (such as input_file) into plain text using Whisper.
'''


    file_context = state.get("input_file", "None")
    sys_msg = SystemMessage(
        content=f"""You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.:


Attached input file (if any): {file_context}

If an input file is attached, determine its type (image/audio/other) and use the appropriate tool to analyze or transcribe its content before answering the question.
You can use any of the following tools. {tool_descriptions}
Remeber, You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
"""

    )

    messages = [sys_msg] + [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in state["messages"]
    ]

    response = llm.invoke(messages)
    final_answer = extract_final_answer(response.content)
    return {
        "messages": [AIMessage(content=final_answer)],
        "input_file": file_context
    }

# Define the graph
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

builder.set_entry_point("assistant")
builder.set_finish_point("assistant")

alfred = builder.compile()

with open("graph_diagram.png", "wb") as f:
    f.write(alfred.get_graph().draw_mermaid_png())

def run_and_submit_all():
    if not HF_USERNAME:
        print("HF_USERNAME is not set in the environment.")
        return

    questions_url = f"{API_URL}/questions"
    submit_url = f"{API_URL}/submit"

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
        input_content = q.get("question", "")
        file_context = ""

        task_id = q.get("task_id")
        file_data = q.get("file_name")
        if isinstance(file_data, str) and file_data:
            try:
                # Construct URL to download file (assuming the API supports it)
                file_url = f"{API_URL}/files/{task_id}"
                
                # print("OK 01", file_url)

                file_response = requests.get(file_url)
                
                # print("OK 0", file_response)


                file_response.raise_for_status()
                # print("OK 1")

                mime_type, _ = mimetypes.guess_type(file_data)
                # print("OK 2", mime_type)
                suffix = mimetypes.guess_extension(mime_type) or os.path.splitext(file_data)[1] or ""

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_response.content)
                    file_context = tmp.name

            except Exception as fe:
                print("Exception is executed.")
                input_content += f"\n\nError loading attached file: {fe}"

        answer = alfred.invoke({
            "messages": [HumanMessage(content=input_content)],
            "input_file": file_context
        })['messages'][-1].content

        print("Question:" + q.get("question", "") + "\n", "Answer:" + answer)

        answers.append({
            "task_id": q["task_id"],
            "submitted_answer": answer,
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
