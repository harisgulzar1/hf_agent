# Implementation of Multi-Input LangGraph Agent and its MCP Implementation

This repository contains a **LangGraph-based multi-input AI agent** that can handle text, images, audio, and video inputs. It uses **OpenAI models** bound to multiple tools for web search, image analysis, audio transcription, and video summarization, executing them dynamically based on the user's query.
Both the simple LangGraph implementation and its MCP Client-Server implementation are done.

### 
![LangGraph Workflow with MCP Implementation](Agent on MCP.jpg)
![LangGraph Workflow](graph_diagram.png)

## Features

- **Dynamic Tool Execution**: The agent intelligently decides when to call tools via `tool_calls`.
- **Multi-Modal Support**:
  - Web search for up-to-date information.
  - Image analysis using OpenAI vision models.
  - Audio transcription using Whisper.
  - Video summarization (YouTube placeholder implementation).
- **Stateful Workflow** with [LangGraph](https://github.com/langchain-ai/langgraph) managing transitions between tool usage and final output.
- **Batch Processing**: Capable of fetching tasks from an API, running them through the agent, and submitting results.

## Tools Used

- [**LangChain**](https://github.com/langchain-ai/langchain) – Framework for building LLM-powered applications.
- [**LangGraph**](https://github.com/langchain-ai/langgraph) – Graph-based orchestration for multi-step reasoning and tool use.
- [**OpenAI Python Client**](https://github.com/openai/openai-python) – For GPT models, vision, and Whisper.
- [**python-dotenv**](https://github.com/theskumar/python-dotenv) – Environment variable management.
- [**pandas**](https://github.com/pandas-dev/pandas) – Data handling utilities.
- [**requests**](https://github.com/psf/requests) – HTTP client for API interaction.
- [**langsmith**](https://github.com/langchain-ai/langsmith-sdk) – Tracing and debugging for LangChain apps.

## Main Workflow

The core workflow follows this structure:
1. **Main Node (`main_node`)**: The LLM is invoked (with tools bound) and may emit `tool_calls`.
2. **Tool Node (`ToolNode`)**: Executes any emitted tool calls.
3. **Conditional Routing**: If further tool calls are needed, return to `main_node`; otherwise, proceed to output.
4. **Output Node (`output_node`)**: Produces the final answer.

This flow is visualized in `graph_diagram.png`.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_dir>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=your_key_here
   HF_USERNAME=your_hf_username
   API_URL=https://your.api.url/
   ```
4. Run the agent:
   ```bash
   python multi_input_agent.py
   ```

## API Batch Mode
The script’s `run_and_submit_all()` function can fetch multiple tasks from a specified API, run them, and submit answers back.

## HuggingFace Course
This GitHub repo is built as the final project for [HuggingFace AI Agents Course.](https://huggingface.co/learn/agents-course/unit0/introduction)

After finishing the course, one can get the official certificate from HuggingFace.
![Course Completion Certificate](certificate.png)


## License
This project is licensed under the Apache License 2.0.
