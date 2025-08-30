#!/usr/bin/env python3
import os
import json
import asyncio
import subprocess
import sys
import inspect
from typing import TypedDict, Annotated, Dict, Any, List
import requests
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers import LangChainTracer

# LangGraph imports
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    task_id: str
    iteration: int

class MCPClientApp:
    def __init__(self, config_path: str = "mcp_client_app.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.app_info = self.config["app_info"]
        self.mcp_server_config = self.config["mcp_server"]
        self.langgraph_config = self.config["langgraph_config"]
        self.api_config = self.config["api_config"]
        self.environment_config = self.config["environment"]
        self.tracing_config = self.config["tracing"]
        
        # Validate environment variables
        self._validate_environment()
        
        # Setup tracing if enabled
        self._setup_tracing()
        
        # Initialize MCP connection
        self.mcp_session = None
        self.stdio_client_cm = None
        self.available_tools = {}
        self.prompts = {}
        
        # Initialize LLMs for each node
        self.llms = {}
        self._initialize_llms()
        
        # Build LangGraph
        self.agent = None

    def _validate_environment(self):
        """Validate required environment variables"""
        missing_vars = []
        for var in self.environment_config["required_vars"]:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _setup_tracing(self):
        """Setup LangChain tracing if enabled"""
        if self.tracing_config.get("enabled", False):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            project_name = os.getenv("LANGCHAIN_PROJECT", self.tracing_config.get("project_name", "MultiInputAgentTrace"))
            os.environ["LANGCHAIN_PROJECT"] = project_name

    def _initialize_llms(self):
        """Initialize LLM instances for each node"""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        callbacks = [LangChainTracer()] if self.tracing_config.get("enabled", False) else []
        
        for node_name, node_config in self.langgraph_config["nodes"].items():
            if node_config.get("type") != "mcp_tool_node":  # Skip tool nodes
                llm_endpoint = node_config.get("llm_endpoint", "gpt-4o")
                self.llms[node_name] = ChatOpenAI(
                    model=llm_endpoint,
                    api_key=openai_api_key,
                    verbose=True,
                    callbacks=callbacks
                )

    async def _start_mcp_server(self):
        """Start the MCP tools server"""
        try:
            server_params = StdioServerParameters(
                command=self.mcp_server_config["command"],
                args=self.mcp_server_config["args"],
                env=self.mcp_server_config.get("env", {})
            )
            
            # Create the stdio client context manager
            self.stdio_client_cm = stdio_client(server_params)
            read_stream, write_stream = await self.stdio_client_cm.__aenter__()
            
            # Create the session
            self.mcp_session = ClientSession(read_stream, write_stream)
            await self.mcp_session.__aenter__()
            
            # Initialize the session
            await self.mcp_session.initialize()
            
            # Load available tools
            tools_result = await self.mcp_session.list_tools()
            self.available_tools = {tool.name: tool for tool in tools_result.tools}
            
            # Load prompts
            resources_result = await self.mcp_session.list_resources()
            
            for resource in resources_result.resources:
                # Convert URI to string if it's an AnyUrl object
                uri_str = str(resource.uri)
                if uri_str.startswith("prompt://"):
                    prompt_name = uri_str.replace("prompt://", "")
                    content_result = await self.mcp_session.read_resource(uri_str)
                    # Extract the actual text content from the ReadResourceResult
                    if hasattr(content_result, 'contents') and content_result.contents:
                        # Get the first content item and extract its text
                        first_content = content_result.contents[0]
                        if hasattr(first_content, 'text'):
                            self.prompts[prompt_name] = first_content.text
                        else:
                            self.prompts[prompt_name] = str(first_content)
                    else:
                        # Fallback: try to convert the whole result to string
                        self.prompts[prompt_name] = str(content_result)
                    
        except Exception as e:
            print(f"Error in _start_mcp_server: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool"""
        if not self.mcp_session or tool_name not in self.available_tools:
            return f"Error: Tool {tool_name} not available"
        
        try:
            result = await self.mcp_session.call_tool(tool_name, arguments)
            # Extract text content from result
            if result.content and len(result.content) > 0:
                first_content = result.content[0]
                if hasattr(first_content, 'text'):
                    return first_content.text
                else:
                    return str(first_content)
            return "No result returned"
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

    def _extract_final_answer(self, content: str) -> str:
        """Extract final answer from content"""
        if "FINAL ANSWER:" in content:
            return content.split("FINAL ANSWER:", 1)[1].strip()
        return content.strip()

    def _should_continue(self, state: AgentState):
        """Route to tools if the LLM's last message includes tool calls; otherwise finish."""
        messages = state["messages"]
        iteration = state.get("iteration", 0)
        max_iterations = self.langgraph_config["nodes"]["handler"]["max_iterations"]
        last = messages[-1]
        
        if getattr(last, "tool_calls", None) and iteration < max_iterations:
            return "tools"
        return "output"

    async def _handler_node(self, state: AgentState) -> AgentState:
        """Handler node implementation"""
        import uuid
        import re
        
        messages = state["messages"]
        task_id = state["task_id"]
        iteration = state.get("iteration", 0)
        
        # Get system prompt from MCP resources
        handler_prompt = self.prompts.get("handler_system_prompt", "You are a helpful assistant.")
        sys_msg = SystemMessage(content=handler_prompt)
        
        # Check if we already have tool results in the conversation
        has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
        
        # If we have tool results, don't call more tools, just proceed to output
        if has_tool_results:
            response = self.llms["handler"].invoke([sys_msg] + messages)
            return {
                "messages": messages + [response],
                "task_id": task_id,
                "iteration": iteration + 1,
            }
        
        # Use the LLM to analyze the question
        response = self.llms["handler"].invoke([sys_msg] + messages)
        
        # Get the original question for analysis
        original_question = messages[0].content if messages else ""
        original_question_lower = original_question.lower()
        
        # Initialize tool_calls list
        tool_calls = []
        
        # 1. Check for YouTube video analysis first (highest priority for video URLs)
        youtube_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+'
        youtube_match = re.search(youtube_pattern, original_question)
        if youtube_match:
            print("🎥 Executing video_analysis_tool")
            tool_calls.append({
                "id": str(uuid.uuid4()),
                "name": "video_analysis_tool",
                "args": {"youtube_url": youtube_match.group(0), "prompt": original_question}
            })
        
        # 2. Check for web search needs (high priority for research questions)
        elif any(indicator in original_question_lower for indicator in 
                ["wikipedia", "search the web", "find information", "look up", "research", "how many", "what is"]):
            print("🔍 Executing openai_web_search")
            tool_calls.append({
                "id": str(uuid.uuid4()),
                "name": "openai_web_search",
                "args": {"query": original_question}
            })
        
        # 3. Check for file-based tasks only if task_id is present and in the message
        elif task_id and "task_id:" in original_question:
            # Image analysis - check for image-related keywords
            if any(indicator in original_question_lower for indicator in 
                  ["image", "picture", "photo", "visual", "see", "shown", "diagram", "chart"]):
                print("🖼️ Executing image_analyzer_tool")
                # Extract the actual question without task_id for better analysis
                question_without_task = original_question.split("task_id:")[0].strip()
                tool_calls.append({
                    "id": str(uuid.uuid4()),
                    "name": "image_analyzer_tool", 
                    "args": {"task_id": task_id, "prompt": question_without_task}
                })
            
            # Audio transcription - check for audio-related keywords
            elif any(indicator in original_question_lower for indicator in 
                    ["audio", "sound", "speech", "voice", "transcribe", "listen", "hear"]):
                print("🎵 Executing audio_transcription_tool")
                question_without_task = original_question.split("task_id:")[0].strip()
                tool_calls.append({
                    "id": str(uuid.uuid4()),
                    "name": "audio_transcription_tool",
                    "args": {"task_id": task_id, "prompt": question_without_task}
                })
            
            # Code compilation - check for code-related keywords
            elif any(indicator in original_question_lower for indicator in 
                    ["code", "program", "compile", "run", "execute", "function", "script", "algorithm"]):
                print("💻 Executing compile_code")
                question_without_task = original_question.split("task_id:")[0].strip()
                tool_calls.append({
                    "id": str(uuid.uuid4()),
                    "name": "compile_code",
                    "args": {"task_id": task_id, "message": question_without_task}
                })
        
        # Add tool_calls to the response if any were detected
        if tool_calls:
            response.tool_calls = tool_calls
        
        return {
            "messages": messages + [response],
            "task_id": task_id,
            "iteration": iteration + 1,
        }

    async def _tool_node(self, state: AgentState) -> AgentState:
        """Tool node implementation using MCP"""
        from langchain_core.messages import ToolMessage
        
        messages = state["messages"]
        last_message = messages[-1]
        
        # Extract tool calls from the last message
        tool_calls = getattr(last_message, "tool_calls", [])
        
        tool_responses = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
            tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else tool_call.args
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", "unknown")
            
            print(f"🔧 Calling MCP tool: {tool_name} with args: {tool_args}")
            result = await self._call_mcp_tool(tool_name, tool_args)
            print(f"✅ Tool {tool_name} completed")
            
            # Create a ToolMessage with the result
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_id
            )
            tool_responses.append(tool_message)
        
        return {
            "messages": messages + tool_responses,
            "task_id": state["task_id"],
            "iteration": state.get("iteration", 0)
        }

    def _output_node(self, state: AgentState) -> AgentState:
        """Output node implementation"""
        # Get system prompt from MCP resources
        output_prompt = self.prompts.get("output_system_prompt", "Provide a final answer.")
        sys_msg = SystemMessage(content=output_prompt)
        
        messages = state["messages"]
        
        # Convert messages to the format expected by the LLM
        llm_messages = [sys_msg]
        for m in messages:
            if isinstance(m, HumanMessage):
                llm_messages.append(HumanMessage(content=m.content))
            elif isinstance(m, AIMessage):
                llm_messages.append(AIMessage(content=m.content))
        
        response = self.llms["output"].invoke(llm_messages)
        final_answer = self._extract_final_answer(response.content)
        
        return {
            "messages": [AIMessage(content=f"FINAL ANSWER: {final_answer}")],
            "task_id": state["task_id"],
            "iteration": state.get("iteration", 0)
        }

    async def _build_agent(self):
        """Build the LangGraph agent"""
        # Define the graph
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("handler", self._handler_node)
        builder.add_node("output", self._output_node) 
        builder.add_node("tools", self._tool_node)
        
        # Add edges based on configuration
        edges_config = self.langgraph_config["edges"]
        
        builder.add_edge(START, "handler")
        builder.add_conditional_edges("handler", self._should_continue, ["output", "tools"])
        builder.add_edge("tools", "handler")
        builder.add_edge("output", END)
        
        self.agent = builder.compile()

    async def process_question(self, question: str, task_id: str = None) -> str:
        """Process a single question using the agent"""
        # Prepare the initial message - only include task_id if there's actually a file
        # Check if this question needs a file by looking for file-related keywords
        question_lower = question.lower()
        file_indicators = [
            "image", "picture", "photo", "audio", "sound", "code", "file", 
            "attachment", "attached", "upload", "analyze", "transcribe"
        ]
        
        # Only include task_id in message if there are file indicators AND task_id exists
        if task_id and any(indicator in question_lower for indicator in file_indicators):
            message_content = f"{question}\ntask_id: {task_id}"
        else:
            message_content = question
        
        initial_message = [HumanMessage(content=message_content)]
        
        # Run the agent
        result = await self.agent.ainvoke({
            "messages": initial_message,
            "task_id": task_id or "",
            "iteration": 0
        })
        
        # Extract the final answer
        final_message = result['messages'][-1].content
        if final_message.strip().startswith("FINAL ANSWER:"):
            answer = final_message.strip()[len("FINAL ANSWER:"):].strip()
        else:
            answer = final_message.strip()
        
        return answer

    async def run_and_submit_all(self):
        """Run agent on all questions and submit results"""
        hf_username = os.getenv("HF_USERNAME")
        if not hf_username:
            print("HF_USERNAME is not set in the environment.")
            return

        api_url = os.getenv("API_URL", "https://jofthomas-unit4-scoring.hf.space/")
        questions_url = f"{api_url.rstrip('/')}{self.api_config['questions_endpoint']}"
        submit_url = f"{api_url.rstrip('/')}{self.api_config['submit_endpoint']}"
        timeout = self.api_config.get("timeout", 15)

        # Get agent code for submission
        agent_code = inspect.getsource(AgentState)

        try:
            response = requests.get(questions_url, timeout=timeout)
            response.raise_for_status()
            questions = response.json()
        except Exception as e:
            print("Error fetching questions:", e)
            return

        answers = []
        for q in questions:
            input_content = q.get("question", "")
            task_id = q.get("task_id")
            
            print(f"Processing question: {input_content}")
            
            try:
                answer = await self.process_question(input_content, task_id)
                print(f"Answer: {answer}")
                
                answers.append({
                    "task_id": q["task_id"],
                    "submitted_answer": answer,
                })
            except Exception as e:
                print(f"Error processing question {q.get('task_id', 'unknown')}: {e}")
                answers.append({
                    "task_id": q["task_id"],
                    "submitted_answer": "Error processing question",
                })

        # Submit answers
        submission = {
            "username": hf_username,
            "agent_code": agent_code,
            "answers": answers
        }
        
        try:
            res = requests.post(submit_url, json=submission, timeout=timeout)
            res.raise_for_status()
            result = res.json()
            print("Submission Result:", result)
        except Exception as e:
            print("Submission failed:", e)

    async def run(self):
        """Main run method"""
        try:
            # Start MCP server
            print("Starting MCP server...")
            await self._start_mcp_server()
            print("MCP server started successfully")
            
            # Build agent
            print("Building LangGraph agent...")
            await self._build_agent()
            print("Agent built successfully")
            
            # Process questions
            print("Processing questions...")
            await self.run_and_submit_all()
            
        except Exception as e:
            print(f"Error in main run: {e}")
        finally:
            # Clean up MCP session
            if self.mcp_session:
                try:
                    await self.mcp_session.__aexit__(None, None, None)
                except:
                    pass
            if self.stdio_client_cm:
                try:
                    await self.stdio_client_cm.__aexit__(None, None, None)
                except:
                    pass

async def main():
    """Main entry point"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "mcp_client_app.json"
    app = MCPClientApp(config_path)
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())