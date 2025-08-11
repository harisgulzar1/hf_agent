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
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from langchain.callbacks.tracers import LangChainTracer
import os


# Load environment variables from .env file
load_dotenv()

# Optionally set the project name programmatically
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MultiInputAgentTrace")

# Get values
HF_USERNAME = os.getenv("HF_USERNAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = os.getenv("API_URL", "https://jofthomas-unit4-scoring.hf.space/")

# OpenAI native client for tool use
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@tool
def openai_web_search(query: str) -> str:
    """
    Perform a web search using OpenAI's web_search_preview tool.

    Args:
        query (str): The search query string.

    Returns:
        str: A summarized result from the web search.
    """
    print("✅ WEB TOOL EXECUTED")
    response = openai_client.responses.create(
        model="gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input=query
    )
    return response.output_text


@tool
def image_analyzer_tool(
    task_id: str,
    prompt: str = "Describe this image.",
) -> str:
    """
    Analyze an image retrived by the corresponding task_id:
    task_id is the one given as argument, not the file name present in the prompt message. task_id is a sequence of alphanumeric characters e.g. 99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.

    Args:
    task_id (str): Task ID, used to download the corresponding file realted to the task. GET {API_URL}/files/{task_id}
    prompt: prompt to describe the image.
    """
    print("✅ IMAGE TOOL EXECUTED")

    def _download_from_backend(task_id: str) -> tuple[str, str]:
        url = f"{API_URL.rstrip('/')}/files/{task_id}"
        try:
            with requests.get(url, timeout=30, stream=True) as resp:
                resp.raise_for_status()
                # Prefer filename from Content-Disposition; infer type from Content-Type
                cd = resp.headers.get("Content-Disposition", "")
                ctype = resp.headers.get("Content-Type", "image/png")
                if "png" in ctype:
                    default_ext = ".png"
                elif ("jpeg" in ctype) or ("jpg" in ctype):
                    default_ext = ".jpg"
                elif "webp" in ctype:
                    default_ext = ".webp"
                else:
                    default_ext = ".png"

                ext = default_ext
                if "filename=" in cd:
                    fname = cd.split("filename=")[-1].strip('"\' ')
                    ext = os.path.splitext(fname)[1] or default_ext

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                    path = tmp.name

                mime = mimetypes.guess_type(path)[0] or ctype or "image/png"
                return path, mime
        except Exception as e:
            raise RuntimeError(f"download failed from {url}: {e}")

    try:
        path, mime = _download_from_backend(task_id)
        if not os.path.isfile(path):
            return f"Error: Image file not found after download for task_id {task_id}"

        with open(path, "rb") as img_file:
            image_bytes = img_file.read()
        base64_img = base64.b64encode(image_bytes).decode("utf-8")

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who analyzes images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_img}"}},
                    ],
                },
            ],
            max_tokens=1000,
        )
        try:
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error analyzing image: {e}"
    except Exception as e:
        return f"Error preparing image for analysis: {e}"

@tool
def audio_transcription_tool(
    task_id: str,
    prompt: str = "Transcribe this audio.",
) -> str:
    """
    Transcribe an audio file by downloading from the api using task_id.
    task_id is the one given as argument, not the file name present in the prompt message. task_id is a sequence of alphanumeric characters e.g. 99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.

    Args:
        task_id (str): Task ID, used to download the corresponding file realted to the task. GET {API_URL}/files/{task_id}
        prompt (str, optional): Natural language prompt to guide the analysis. Defaults to "Transcribe this audio."

    Returns:
        str: The transcription of the audio generated by Whisper.
    """
    print("✅ AUDIO TOOL EXECUTED")

    def _download_from_backend(task_id: str) -> str:
        url = f"{API_URL.rstrip('/')}/files/{task_id}"
        try:
            with requests.get(url, timeout=30, stream=True) as resp:
                resp.raise_for_status()
                # Try filename from Content-Disposition; else infer from Content-Type
                cd = resp.headers.get("Content-Disposition", "")
                ctype = resp.headers.get("Content-Type", "audio/mpeg")
                ext = ".mp3" if ("mpeg" in ctype or "mp3" in ctype) else ".wav"
                if "filename=" in cd:
                    # naive parse; good enough for typical cases
                    fname = cd.split("filename=")[-1].strip('"\' ')
                    ext = os.path.splitext(fname)[1] or ext
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                    return tmp.name
        except Exception as e:
            raise RuntimeError(f"download failed from {url}: {e}")

    try:
        resolved_path = _download_from_backend(task_id)
        if not os.path.isfile(resolved_path):
            return f"Error: Audio file not found after download for task_id {task_id}"

        with open(resolved_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                prompt=prompt,
            )
        return transcript.strip()
    except Exception as e:
        return f"Error during transcription: {e}"

import re
import yt_dlp  # video download
import cv2  # frame sampling

@tool
def video_analysis_tool(
    youtube_url: str,
    prompt: str = "Transcribe or summarize the video.",
    mode: str = "auto",   # "auto" | "transcribe" | "summarize"
    max_chars: int = 12000,
) -> str:
    """

    Downloads a YouTube video directly from the provided URL, extracts audio for transcription,
    and optionally samples frames for visual analysis without requiring ffmpeg.
    Uses OpenAI Whisper for transcription and GPT-4 Vision for frame analysis.
    
    Args:
        youtube_url (str): Used to download the video.
        prompt (str, optional): Natural language prompt to guide the analysis. Defaults to "Transcribe or summarize the video."

    Returns:
        str: Output the video analysis summary.
    """
    print("✅ VIDEO TOOL EXECUTED")

    # Step 1: Download video using yt-dlp
    tmp_dir = tempfile.mkdtemp(prefix="ytvid_")
    out_path = os.path.join(tmp_dir, "video.mp4")
    ydl_opts = {
        'outtmpl': out_path,
        'format': 'mp4/best',
        'quiet': True,
        'no_warnings': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        video_path = out_path
    except Exception as e:
        return f"Error downloading video from YouTube: {e}"

    # Step 2: Transcribe audio
    try:
        with open(video_path, "rb") as media_file:
            transcript_text = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=media_file,
                response_format="text",
                prompt=prompt,
            )
    except Exception as e:
        return f"Error during transcription: {e}"

    if mode == "transcribe":
        return transcript_text.strip()

    # Step 3: Sample frames for visual analysis
    frame_descriptions = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file for frame analysis."
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = int(max(1, round(5 * fps)))  # sample every ~5 seconds
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = 0
        taken = 0
        while idx < frame_count and taken < 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            ok_png, buf = cv2.imencode(".png", frame)
            if not ok_png:
                idx += step
                continue
            b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")
            try:
                vision_resp = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert visual analyst."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                            ],
                        },
                    ],
                    max_tokens=300,
                    temperature=0.2,
                )
                desc = vision_resp.choices[0].message.content.strip()
                frame_descriptions.append(f"[t={int(idx/fps)}s] {desc}")
                taken += 1
            except Exception as e:
                frame_descriptions.append(f"[frame {idx}] Vision error: {e}")
                taken += 1
            idx += step
        cap.release()
    except Exception as e:
        return f"Error during frame extraction/analysis: {e}"

    # Step 4: Summarize transcript + visuals
    if mode == "summarize" or mode == "auto":
        try:
            visual_context = "\n".join(frame_descriptions)
            text_for_llm = transcript_text[:max_chars] + ("…" if len(transcript_text) > max_chars else "")
            summary_resp = openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You summarize videos using transcript and visual context."},
                    {"role": "user", "content": f"{prompt}\n\nTRANSCRIPT:\n{text_for_llm}\n\nVISUAL CUES:\n{visual_context}"},
                ],
                max_tokens=900,
                temperature=0.2,
            )
            return summary_resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Summary generation failed: {e}\n\nTRANSCRIPT:\n{text_for_llm}\n\nVISUAL CUES:\n{visual_context}"



tools = [openai_web_search, image_analyzer_tool, audio_transcription_tool, video_analysis_tool]

print("Registered tools:", [t.name for t in tools])


llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    verbose=True,
    callbacks=[LangChainTracer()]
).bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task_id: str
    iteration: int


def extract_final_answer(content: str) -> str:
    if "FINAL ANSWER:" in content:
        return content.split("FINAL ANSWER:", 1)[1].strip()
    return content.strip()

# --- Handler node ---
def get_valid_file_path(input_file):
    if isinstance(input_file, list):
        # Find the last item that is a non-empty string and a valid file
        for v in reversed(input_file):
            if isinstance(v, str) and v and os.path.isfile(v):
                return v
        return None
    if isinstance(input_file, str) and input_file and os.path.isfile(input_file):
        return input_file
    return None

# --- Output node ---
@traceable(name="output-node")
def output_node(state: AgentState):
    input_file = state.get("input_file")
    input_file_val = get_valid_file_path(input_file)
    """
    Now that all information and tool results are available, produce the final answer in the required format.
    """
    file_context = input_file_val or "None"

    sys_msg = SystemMessage(
        content=f"""
            You are a general AI assistant. I will ask you a question.
            Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
            If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
            If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
            If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
            Remember: Only output FINAL ANSWER: [YOUR FINAL ANSWER] in this step!
            """
    )
    messages = state["messages"]
    chat_messages = [sys_msg] + [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
    ]
    response = llm.invoke(chat_messages)
    final_answer = extract_final_answer(response.content)
    return {
        "messages": [AIMessage(content=f"FINAL ANSWER: {final_answer}")],
        "input_file": file_context,
        "iteration": state.get("iteration")
    }


# ------------------ Main loop (refactored) ------------------
@traceable(name="condition")
def should_continue(state: AgentState):
    """Route to tools if the LLM's last message includes tool calls; otherwise finish."""
    messages = state["messages"]
    iteration = state.get("iteration", 0)
    last = messages[-1]
    if getattr(last, "tool_calls", None) and iteration < 3:
        return "tools"
    return "output"

@traceable(name="handler")
def handler_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    task_id = state["task_id"]
    iteration = state.get("iteration", 0)

    # Encourage tool use via a light system prompt, but DO NOT craft function_call payloads yourself.

    tool_descriptions = '''
        openai_web_search(query: str) -> str: For web info.
        image_analyzer_tool(task_id: str, prompt: str) -> str: Call this tool for image analysis only when there is an attached image and the prompt is about the image.
        audio_transcription_tool(task_id: str, prompt: str) -> str: For audio to text.
        video_analysis_tool(youtube_url: str, prompt: str) -> str: For YouTube video transcript/summary.
        '''
    
    sys = SystemMessage(content=(        
        "You are a helpful assistant. Use tools when needed."
        + tool_descriptions +
        "If you have enough information to answer, answer concisely."
    ))

    response = llm.invoke([sys, *messages])
    return {
        "messages": messages + [response],
        "task_id": task_id,
        "iteration": iteration + 1,
    }


tool_node = ToolNode(tools=tools)

# Define the graph
builder = StateGraph(AgentState)
builder.add_node("handler", handler_node)
builder.add_node("output", output_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "handler")
builder.add_conditional_edges("handler", should_continue, ["output", "tools"])
builder.add_edge("tools", "handler")
builder.add_edge("output", END)

alfred = builder.compile()

with open("graph_diagram.png", "wb") as f:
    f.write(alfred.get_graph().draw_mermaid_png())

@traceable(name="run_and_submit_all_agent")
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
        task_id = q.get("task_id")
        if q.get("file_name") != "":
            final_message = [HumanMessage(content=input_content+"\ntask_id: "+ task_id)]
        else:
            final_message =  [HumanMessage(content=input_content)]
        
        raw_answer = alfred.invoke({
            "messages": final_message,
            "task_id": task_id,
            "iteration": 0
        })['messages'][-1].content
        # Remove leading "FINAL ANSWER:" if present, and strip whitespace
        if raw_answer.strip().startswith("FINAL ANSWER:"):
            answer = raw_answer.strip()[len("FINAL ANSWER:"):].strip()
        else:
            answer = raw_answer.strip()
        print("Question:" + q.get("question", "") + "\n" + "Answer:" + answer)
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

import langsmith

if __name__ == "__main__":
    # Enable fine-grained tracing for LangGraph and LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "MultiInputAgentTrace")
    run_and_submit_all()
