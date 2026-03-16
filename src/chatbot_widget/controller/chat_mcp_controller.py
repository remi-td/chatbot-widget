import asyncio
import json
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from chatbot_widget.ui.chat_view import ChatView 

from chatbot_widget.utils.utils import run_async
from chatbot_widget.mcp.server_manager import MCPServerManager

class ChatMCPController:

    def __init__(self, mcp_server_manager: MCPServerManager, model: str = "openai:gpt-5", system_prompt: str | None = None):
        """Initialize the MCP chat controller.

        Args:
            mcp_server_manager: Manager that coordinates available MCP servers.
            model: Identifier for the chat model. Currently only OpenAI models are supported
                (for example: openai:gpt-4o-mini, openai:gpt-4o, openai:gpt-4.1-mini, openai:gpt-5).
            system_prompt: Optional system prompt to set the assistant's behavior.

        Raises:
            ValueError: If the provided model is not an OpenAI model.
        """
        self.mcp = mcp_server_manager
        if not model.startswith("openai:"):
            raise ValueError(
                f"Only OpenAI chat interfaces are supported for now. Received '{model}'."
            )
        self.model = model

        self.lookup_tool_server = self.mcp.get_tool_server_dict()
        self.server_port_dict = self.mcp.get_server_port_dict()

        client_dict = {
            server: {
                "transport": "streamable_http",
                "url": f"http://localhost:{port}/mcp",
            }
            for server, port in self.server_port_dict.items()
        }
        self.client = MultiServerMCPClient(client_dict)

        self.ui = ChatView()

        all_tools = run_async(self.client.get_tools())
        self.agent = create_agent(self.model, all_tools, system_prompt=system_prompt, checkpointer=InMemorySaver())

        # connect UI
        self.ui.on_send(self.handle_input)


    def display(self):
        self.ui.display()

    # ---------------------------------------------------------------
    # Main entrypoint
    # ---------------------------------------------------------------

    def handle_input(self, msg):
        if not isinstance(msg, str):
            msg = str(msg)
        if not msg.strip():
            return

        # special command
        if msg.startswith("/"):
            reply = self._execute_command(msg.strip())
            self.ui.receive_message(reply)
            return

        self.ui.set_busy(True)
        self.ui.show_waiting_indicator()

        async def _stream():
            # run_id -> (stream_id, accumulated_text)
            model_streams: dict[str, tuple[str, str]] = {}
            # run_id -> last UI update timestamp
            last_update: dict[str, float] = {}
            INTERVAL = 1.0  # seconds between UI refreshes during streaming

            async for event in self.agent.astream_events(
                {"messages": [{"role": "user", "content": msg}]},
                {"configurable": {"thread_id": "1"}},
                version="v2",
            ):
                kind = event["event"]
                run_id = event.get("run_id", "")

                if kind == "on_tool_start":
                    tool_name = event["name"]
                    args = event["data"].get("input", {})
                    server_name = self.lookup_tool_server.get(tool_name)
                    full_tool_name = f"{server_name}::{tool_name}"
                    try:
                        serialized = json.dumps(args, ensure_ascii=False, default=str)
                    except Exception:
                        serialized = str(args)
                    self.ui.receive_tool_call(run_id, full_tool_name, serialized)

                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    output = event["data"].get("output")
                    if hasattr(output, "content"):
                        content = output.content
                        if isinstance(content, list):
                            content = "\n".join(
                                b.get("text", "") if isinstance(b, dict) else str(b)
                                for b in content
                            )
                    else:
                        content = str(output)
                    self.ui.receive_tool_reply(run_id, tool_name, content)

                elif kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    token = ""
                    if chunk:
                        c = chunk.content
                        if isinstance(c, str):
                            token = c
                        elif isinstance(c, list):
                            token = "".join(b.get("text", "") if isinstance(b, dict) else "" for b in c)
                    if token:
                        if run_id not in model_streams:
                            stream_id = self.ui.start_stream("bot")
                            model_streams[run_id] = (stream_id, token)
                            last_update[run_id] = time.monotonic()
                            self.ui.stream_update(stream_id, token)
                        else:
                            stream_id, accumulated = model_streams[run_id]
                            accumulated += token
                            model_streams[run_id] = (stream_id, accumulated)
                            now = time.monotonic()
                            if now - last_update[run_id] >= INTERVAL:
                                self.ui.stream_update(stream_id, accumulated)
                                last_update[run_id] = now

                elif kind == "on_chat_model_end":
                    output = event["data"].get("output")
                    has_tool_calls = bool(getattr(output, "tool_calls", None))
                    stream_entry = model_streams.pop(run_id, None)
                    last_update.pop(run_id, None)

                    if stream_entry:
                        stream_id, accumulated = stream_entry
                        if has_tool_calls:
                            # intermediate tool-calling step — discard the partial bubble
                            bubble = self.ui._active_streams.pop(stream_id, None)
                            if bubble:
                                children = list(self.ui.chat_box.children)
                                if bubble.widget in children:
                                    children.remove(bubble.widget)
                                    self.ui.chat_box.children = tuple(children)
                        else:
                            # flush any buffered tokens before finalizing
                            self.ui.stream_update(stream_id, accumulated)
                            self.ui.end_stream(stream_id)
                    elif not has_tool_calls and output and output.content:
                        # no streaming happened (e.g. model returned full response at once)
                        c = output.content
                        if isinstance(c, list):
                            c = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
                        self.ui.receive_message(c, "bot")

        try:
            run_async(_stream())
        except Exception as e:
            self.ui.receive_message(f"⚠️ Error: {e}", "bot")
        finally:
            self.ui.set_busy(False)
      
    # ---------------------------------------------------------------
    # Command handler
    # ---------------------------------------------------------------
    def _execute_command(self, msg: str):
        """Parse and execute slash commands like /help, /context, etc."""
        parts = msg.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/help":
            return self.__command_help()
        #elif cmd == "/context":
        #    return self.__command_context()
        elif cmd == "/clear":
            return self.__command_clear()
        elif cmd == "/servers":
            return self.__command_servers()
        #elif cmd == "/logs":
        #    return self.__command_logs(*args)
        elif cmd == "/tools":
            return self.__command_tools()
        #elif cmd == "/testtool":
        #    return self.__command_testtool(*args)
        elif cmd == "/checkall":
            return self.__command_checkall()
        else:
            return f"⚠️ Unknown command: {cmd}. Type /help for a list of commands."


    # -----------------------------
    # Command implementations
    # -----------------------------

    def __command_help(self):
        print("command triggered")
        cmds = {
            "/help": "Show this help message",
            #"/context": "Show current conversation context",
            "/clear": "Clear conversation history",
            "/servers": "List running MCP servers",
            #"/logs <server> [n]": "Show last n log lines for a server",
            "/tools": "List all available MCP tools",
            #"/testtool <server> <tool> [args_json]": "Call an MCP tool directly",
            "/checkall": "Check health of all servers",
        }
        return "\n".join([f"- **{k}** {v}" for k, v in cmds.items()])

    def __command_context(self):
        return "🧠 Context inspection not yet implemented."


    def __command_clear(self):
        self.agent.checkpointer = InMemorySaver()
        return "🧹 Conversation history cleared."



    def __command_servers(self):
        servers = self.mcp.get_server_port_dict()
        if not servers:
            return "No active MCP servers."
        return "\n".join([f"{name}: port {port}" for name, port in servers.items()])


    def __command_logs(self, server_name=None, n="50"):
        if not server_name:
            return "Usage: /logs <server_name> [n]"
        try:
            n = int(n)
        except ValueError:
            n = 50
        logs = self.mcp.show_logs(server_name, n)
        return logs or f"No logs found for {server_name}."


    def __command_tools(self):
        tools = self.mcp.get_tool_server_dict()
        if not tools:
            return "No tools available."
        return "\n\n".join([f"{s} :: {t}" for t, s in tools.items()])


    def __command_testtool(self, server_name=None, tool_name=None, args_json="{}"):
        if not (server_name and tool_name):
            return "Usage: /testtool <server_name> <tool_name> [args_json]"
        try:
            args = json.loads(args_json)
        except Exception:
            return "⚠️ Invalid JSON for args."
        result = self.mcp.test_tool(server_name, tool_name, args)
        return json.dumps(result, indent=2) if result else "No result returned."


    def __command_checkall(self):
        results = self.mcp.check_all()
        return results or "Checked all servers."
