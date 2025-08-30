# Background

I have implemented a Langgraph agent which which has three nodes i.e. handler_node, tool_node, and output_node.
Inside tool_node, there are following tools which are implemented in form of python functions under the @tool decorator.
'compile_code', 'openai_web_search', 'image_analyzer_tool', 'audio_transcription_tool', 'video_analysis_tool'
The main code, takes the questions through a run_and_submit_all function and passes the questions along with the attached file (if any) to the agent by using the AgentState object.


# Instruction
All of the code is implemented in one python file, but I want to make it more scalable by transforming the implementation of tools and prompts in terms of an MCP server.

I want to create following sections of the code:

## MCP Tools Server

This section contains two files:
1. mcp_tools_server.json
This files contains the configuration of each tools that are to be implemented.
The configuration will include the tool name, the specific model end point for each tool, and description of each tool used as docstring including the details of the arguments.
As in the main code the system prompts are used for each node in the Langgraph. These prompts should also be modeled as mcp server resouces. The MCP client should be able to fetch these prompts to insert as system prompts of the Langgraph nodes.

2. mcp_tools_server.py
This file reads the configuration information from the mcp_tools_server.json and combiles are runnable mcp server. After running this file, it deploys the tools as mcp server ready to be accessed by mcp client.

## MCP Tools Client

This section contains two files:
1. mcp_client_app.json
This files contains the configuration for the main Langgraph agent i.e. the nodes implementation, the LLM endpoints for each node.

2. mcp_client_app.py
This file reads the configuration information from the mcp_client_app.json and combiles are runnable mcp client which is also the main app code. After running this file, it should be able to connect with the mcp server deployed by mcp_tools_server.py

The unified code which i to be modified is provided in attached app.py file.
