async def _analyze_video(self, youtube_url: str, prompt: str = "Transcribe or summarize the video.", 
                           mode: str = "auto", max_chars: int = 12000) -> str:
        """Video analysis tool implementation"""
        print(f"🔧 [MCP Server] Executing video_analysis with URL: {youtube_url}")
        print(f"📝 Prompt: {prompt[:100]}...")
        try:
            # Download video using yt-dlp
            tmp_dir = tempfile.mkdtemp(prefix="ytvid_")
            out_path = os.path.join(tmp_dir, "video.mp4")
            ydl_opts = {
                'outtmpl': out_path,
                'format': 'mp4/best',
                'quiet': True,
                'no_warnings': True
            }
            
            print(f"⬇️ Downloading video from: {youtube_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            # Transcribe audio
            print(f"🎵 Transcribing audio...")
            with open(out_path, "rb") as media_file:
                transcript_text = self.openai_client.audio.transcriptions.create(
                    model=self.tools_config["video_analysis_tool"]["model_endpoints"]["transcription"],
                    file=media_file,
                    response_format="text",
                    prompt=prompt,
                )
            
            if mode == "transcribe":
                print(f"✅ [MCP Server] video_analysis (transcribe) completed")
                return transcript_text.strip()
            
            # Sample frames for visual analysis
            print(f"🖼️ Analyzing video frames...")
            frame_descriptions = []
            cap = cv2.VideoCapture(out_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                interval = self.app_config.get("video_frame_sample_interval", 5)
                step = int(max(1, round(interval * fps)))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                max_frames = self.app_config.get("max_video_frames", 10)
                
                idx = 0
                taken = 0
                while idx < frame_count and taken < max_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    
                    ok_png, buf = cv2.imencode(".png", frame)
                    if ok_png:
                        b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")
                        try:
                            vision_resp = self.openai_client.chat.completions.create(
                                model=self.tools_config["video_analysis_tool"]["model_endpoints"]["vision"],
                                messages=[
                                    {"role": "system", "content": self.resources_config["prompts"]["vision_analysis_system_prompt"]["content"]},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": prompt},
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                                        ],
                                    },
                                ],
                                max_completion_tokens=self.tools_config["video_analysis_tool"]["vision_max_tokens"],
                            )
                            desc = vision_resp.choices[0].message.content.strip()
                            frame_descriptions.append(f"[t={int(idx/fps)}s] {desc}")
                            taken += 1
                        except Exception as e:
                            frame_descriptions.append(f"[frame {idx}] Vision error: {e}")
                            taken += 1
                    idx += step
                cap.release()
            
            # Generate summary if needed
            if mode == "summarize" or mode == "auto":
                print(f"📝 Generating summary...")
                visual_context = "\n".join(frame_descriptions)
                text_for_llm = transcript_text[:max_chars] + ("…" if len(transcript_text) > max_chars else "")
                
                summary_resp = self.openai_client.chat.completions.create(
                    model=self.tools_config["video_analysis_tool"]["model_endpoints"]["summary"],
                    messages=[
                        {"role": "system", "content": self.resources_config["prompts"]["video_summary_system_prompt"]["content"]},
                        {"role": "user", "content": f"{prompt}\n\nTRANSCRIPT:\n{text_for_llm}\n\nVISUAL CUES:\n{visual_context}"},
                    ],
                    max_tokens=self.tools_config["video_analysis_tool"]["summary_max_tokens"],
                )
                result = summary_resp.choices[0].message.content.strip()
                print(f"✅ [MCP Server] video_analysis (summary) completed")
                return result
            
            result = f"TRANSCRIPT:\n{transcript_text}\n\nVISUAL CUES:\n{visual_context}"
            print(f"✅ [MCP Server] video_analysis completed")
            return result
            
        except Exception as e:
            print(f"❌ [MCP Server] Error in video analysis: {e}")
            return f"Error in video analysis: {e}"#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import tempfile
import base64
import mimetypes
from typing import Any, Sequence
import requests
import cv2
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    CallToolRequest, ListResourcesRequest, ListToolsRequest, ReadResourceRequest,
    ServerCapabilities, ResourcesCapability, ToolsCapability
)

# Load environment variables
load_dotenv()

class MCPToolsServer:
    def __init__(self, config_path: str = "mcp_tools_server.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.server_info = self.config["server_info"]
        self.tools_config = self.config["tools"]
        self.resources_config = self.config["resources"]
        self.app_config = self.config["configuration"]
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY") or self.app_config.get("openai_api_key", "").replace("${OPENAI_API_KEY}", "")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Set API URL
        self.api_url = os.getenv("API_URL", self.app_config.get("api_url", "https://jofthomas-unit4-scoring.hf.space/"))
        if self.api_url.startswith("${"):
            self.api_url = "https://jofthomas-unit4-scoring.hf.space/"
        
        self.server = Server(self.server_info["name"])
        self._setup_handlers()

    def _setup_handlers(self):
        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available prompt resources"""
            resources = []
            for prompt_key, prompt_config in self.resources_config["prompts"].items():
                resources.append(Resource(
                    uri=f"prompt://{prompt_key}",
                    name=prompt_config["name"],
                    description=prompt_config["description"],
                    mimeType="text/plain"
                ))
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read prompt resource content"""
            # Convert URI to string if it's an AnyUrl object
            uri_str = str(uri)
            
            if not uri_str.startswith("prompt://"):
                raise ValueError(f"Unsupported resource URI: {uri_str}")
            
            prompt_key = uri_str.replace("prompt://", "")
            if prompt_key not in self.resources_config["prompts"]:
                raise ValueError(f"Unknown prompt resource: {prompt_key}")
            
            return self.resources_config["prompts"][prompt_key]["content"]

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools"""
            tools = []
            for tool_name, tool_config in self.tools_config.items():
                # Convert parameters to MCP format
                parameters = {"type": "object", "properties": {}, "required": []}
                
                for param_name, param_config in tool_config["parameters"].items():
                    parameters["properties"][param_name] = {
                        "type": param_config["type"],
                        "description": param_config["description"]
                    }
                    if param_config.get("required", False):
                        parameters["required"].append(param_name)

                tools.append(Tool(
                    name=tool_name,
                    description=tool_config["description"],
                    inputSchema=parameters
                ))
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls"""
            if name not in self.tools_config:
                raise ValueError(f"Unknown tool: {name}")
            
            try:
                if name == "compile_code":
                    result = await self._compile_code(**arguments)
                elif name == "openai_web_search":
                    result = await self._web_search(**arguments)
                elif name == "image_analyzer_tool":
                    result = await self._analyze_image(**arguments)
                elif name == "audio_transcription_tool":
                    result = await self._transcribe_audio(**arguments)
                elif name == "video_analysis_tool":
                    result = await self._analyze_video(**arguments)
                else:
                    raise ValueError(f"Tool {name} not implemented")
                
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error in {name}: {str(e)}")]

    def _download_from_backend(self, task_id: str) -> tuple[str, str]:
        """Download file from backend API"""
        url = f"{self.api_url.rstrip('/')}/files/{task_id}"
        try:
            timeout = self.app_config.get("download_timeout", 30)
            with requests.get(url, timeout=timeout, stream=True) as resp:
                resp.raise_for_status()
                
                # Get filename and content type
                cd = resp.headers.get("Content-Disposition", "")
                ctype = resp.headers.get("Content-Type", "")
                
                filename = "file"
                if "filename=" in cd:
                    filename = cd.split("filename=")[-1].strip('"\' ')
                
                # Determine file extension
                ext = os.path.splitext(filename)[1] or ""
                if not ext and ctype:
                    if "png" in ctype:
                        ext = ".png"
                    elif "jpeg" in ctype or "jpg" in ctype:
                        ext = ".jpg"
                    elif "mp3" in ctype or "mpeg" in ctype:
                        ext = ".mp3"
                    elif "wav" in ctype:
                        ext = ".wav"
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                    return tmp.name, filename
        except Exception as e:
            raise RuntimeError(f"Download failed from {url}: {e}")

    async def _compile_code(self, task_id: str, message: str = "Compile the attached code and report the result.") -> str:
        """Compile code tool implementation"""
        print(f"🔧 [MCP Server] Executing compile_code with task_id: {task_id}")
        try:
            # Download the code file
            code_path, filename = self._download_from_backend(task_id)
            print(f"📁 Downloaded file: {filename}")
            
            # Read the code content
            with open(code_path, "r", encoding="utf-8", errors="replace") as f:
                code_content = f.read()
            
            # Prepare payload for OpenAI
            user_payload = f"{message}\n\nFilename: {filename}\n---\n{code_content}"
            
            # Call OpenAI with code interpreter
            resp = self.openai_client.responses.create(
                model=self.tools_config["compile_code"]["model_endpoint"],
                input=user_payload,
                tools=self.tools_config["compile_code"]["tools_required"]
            )
            
            result = (getattr(resp, "output_text", "") or "").strip() or "(No output returned by the execution environment.)"
            print(f"✅ [MCP Server] compile_code completed")
            return result
        except Exception as e:
            print(f"❌ [MCP Server] Error in compile_code: {e}")
            return f"Error in compile_code: {e}"

    async def _web_search(self, query: str) -> str:
        """Web search tool implementation"""
        print(f"🔧 [MCP Server] Executing web_search with query: {query[:100]}...")
        try:
            response = self.openai_client.responses.create(
                model=self.tools_config["openai_web_search"]["model_endpoint"],
                tools=self.tools_config["openai_web_search"]["tools_required"],
                input=query
            )
            result = response.output_text
            print(f"✅ [MCP Server] web_search completed")
            return result
        except Exception as e:
            print(f"❌ [MCP Server] Error in web search: {e}")
            return f"Error in web search: {e}"

    async def _analyze_image(self, task_id: str, prompt: str = "Describe this image in detail.") -> str:
        """Image analysis tool implementation"""
        print(f"🔧 [MCP Server] Executing image_analysis with task_id: {task_id}")
        print(f"📝 Prompt: {prompt[:100]}...")
        try:
            # Download image
            path, filename = self._download_from_backend(task_id)
            print(f"📁 Downloaded image: {filename}")
            
            if not os.path.isfile(path):
                return f"Error: Image file not found after download for task_id {task_id}"
            
            # Read and encode image
            with open(path, "rb") as img_file:
                image_bytes = img_file.read()
            base64_img = base64.b64encode(image_bytes).decode("utf-8")
            
            # Get MIME type
            mime = mimetypes.guess_type(path)[0] or "image/png"
            
            # Analyze with GPT-4 Vision
            response = self.openai_client.chat.completions.create(
                model=self.tools_config["image_analyzer_tool"]["model_endpoint"],
                messages=[
                    {"role": "system", "content": self.resources_config["prompts"]["image_analysis_system_prompt"]["content"]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_img}"}},
                        ],
                    },
                ],
                max_tokens=self.tools_config["image_analyzer_tool"].get("max_tokens", 1000),
            )
            
            result = response.choices[0].message.content.strip()
            print(f"✅ [MCP Server] image_analysis completed")
            return result
        except Exception as e:
            print(f"❌ [MCP Server] Error in image analysis: {e}")
            return f"Error in image analysis: {e}"

    async def _transcribe_audio(self, task_id: str, prompt: str = "Transcribe this audio.") -> str:
        """Audio transcription tool implementation"""
        print(f"🔧 [MCP Server] Executing audio_transcription with task_id: {task_id}")
        print(f"📝 Prompt: {prompt[:100]}...")
        try:
            # Download audio file
            resolved_path, filename = self._download_from_backend(task_id)
            print(f"📁 Downloaded audio: {filename}")
            
            if not os.path.isfile(resolved_path):
                return f"Error: Audio file not found after download for task_id {task_id}"
            
            # Transcribe with Whisper
            with open(resolved_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model=self.tools_config["audio_transcription_tool"]["model_endpoint"],
                    file=audio_file,
                    response_format=self.tools_config["audio_transcription_tool"]["response_format"],
                    prompt=prompt,
                )
            result = transcript.strip()
            print(f"✅ [MCP Server] audio_transcription completed")
            return result
        except Exception as e:
            print(f"❌ [MCP Server] Error in audio transcription: {e}")
            return f"Error in audio transcription: {e}"

    async def _analyze_video(self, youtube_url: str, prompt: str = "Transcribe or summarize the video.", 
                           mode: str = "auto", max_chars: int = 12000) -> str:
        """Video analysis tool implementation"""
        try:
            # Download video using yt-dlp
            tmp_dir = tempfile.mkdtemp(prefix="ytvid_")
            out_path = os.path.join(tmp_dir, "video.mp4")
            ydl_opts = {
                'outtmpl': out_path,
                'format': 'mp4/best',
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            # Transcribe audio
            with open(out_path, "rb") as media_file:
                transcript_text = self.openai_client.audio.transcriptions.create(
                    model=self.tools_config["video_analysis_tool"]["model_endpoints"]["transcription"],
                    file=media_file,
                    response_format="text",
                    prompt=prompt,
                )
            
            if mode == "transcribe":
                return transcript_text.strip()
            
            # Sample frames for visual analysis
            frame_descriptions = []
            cap = cv2.VideoCapture(out_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                interval = self.app_config.get("video_frame_sample_interval", 5)
                step = int(max(1, round(interval * fps)))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                max_frames = self.app_config.get("max_video_frames", 10)
                
                idx = 0
                taken = 0
                while idx < frame_count and taken < max_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    
                    ok_png, buf = cv2.imencode(".png", frame)
                    if ok_png:
                        b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")
                        try:
                            vision_resp = self.openai_client.chat.completions.create(
                                model=self.tools_config["video_analysis_tool"]["model_endpoints"]["vision"],
                                messages=[
                                    {"role": "system", "content": self.resources_config["prompts"]["vision_analysis_system_prompt"]["content"]},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": prompt},
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                                        ],
                                    },
                                ],
                                max_completion_tokens=self.tools_config["video_analysis_tool"]["vision_max_tokens"],
                            )
                            desc = vision_resp.choices[0].message.content.strip()
                            frame_descriptions.append(f"[t={int(idx/fps)}s] {desc}")
                            taken += 1
                        except Exception as e:
                            frame_descriptions.append(f"[frame {idx}] Vision error: {e}")
                            taken += 1
                    idx += step
                cap.release()
            
            # Generate summary if needed
            if mode == "summarize" or mode == "auto":
                visual_context = "\n".join(frame_descriptions)
                text_for_llm = transcript_text[:max_chars] + ("…" if len(transcript_text) > max_chars else "")
                
                summary_resp = self.openai_client.chat.completions.create(
                    model=self.tools_config["video_analysis_tool"]["model_endpoints"]["summary"],
                    messages=[
                        {"role": "system", "content": self.resources_config["prompts"]["video_summary_system_prompt"]["content"]},
                        {"role": "user", "content": f"{prompt}\n\nTRANSCRIPT:\n{text_for_llm}\n\nVISUAL CUES:\n{visual_context}"},
                    ],
                    max_tokens=self.tools_config["video_analysis_tool"]["summary_max_tokens"],
                )
                return summary_resp.choices[0].message.content.strip()
            
            return f"TRANSCRIPT:\n{transcript_text}\n\nVISUAL CUES:\n{visual_context}"
            
        except Exception as e:
            return f"Error in video analysis: {e}"

    async def run(self):
        """Run the MCP server"""
        from mcp.types import ServerCapabilities, ResourcesCapability, ToolsCapability
        
        # Define server capabilities
        capabilities = ServerCapabilities(
            resources=ResourcesCapability(subscribe=True, listChanged=True),
            tools=ToolsCapability(listChanged=True)
        )
        
        async with stdio_server() as (read_stream, write_stream):
            init_options = InitializationOptions(
                server_name=self.server_info["name"],
                server_version=self.server_info["version"],
                capabilities=capabilities
            )
            await self.server.run(read_stream, write_stream, init_options)

def main():
    """Main entry point"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "mcp_tools_server.json"
    server = MCPToolsServer(config_path)
    asyncio.run(server.run())

if __name__ == "__main__":
    main()