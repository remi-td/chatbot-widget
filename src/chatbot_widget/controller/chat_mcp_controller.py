import asyncio
import json
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
        self._seen_msgs = 0  # message counter

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

    def handle_input(self, msg: str):
        if not msg.strip():
            return

        # special command
        if msg.startswith("/"):
            reply = self._execute_command(msg.strip())
            self.ui.receive_message(reply)
            return

        self.ui.set_busy(True)
        self.ui.show_waiting_indicator()

        try:
            print("trying send ", msg)
            result = run_async(
                self.agent.ainvoke(
                    {"messages": [{"role": "user", "content": msg}]},
                    {"configurable": {"thread_id": "1"}},
                )
            )

            # only handle new messages
            new_msgs = result["messages"][self._seen_msgs:]
            self._seen_msgs = len(result["messages"])

            for m in new_msgs:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for call in m.tool_calls:
                        tool_name = call["name"]
                        call_id = (
                            call.get("id")
                            if isinstance(call, dict)
                            else getattr(call, "id", None)
                        )
                        if not call_id:
                            call_id = tool_name or "tool_call"
                        call_id = str(call_id)
                        server_name = self.lookup_tool_server.get(tool_name)
                        full_tool_name = f"{server_name}::{tool_name}"
                        tool_args = str(call["args"])
                        self.ui.receive_tool_call(call_id, full_tool_name, tool_args)
                elif getattr(m, "name", None) and m.__class__.__name__ == "ToolMessage":
                    tool_name = getattr(m, "tool_name", None)
                    call_id = (
                        getattr(m, "tool_call_id", None)
                        or getattr(m, "id", None)
                        or getattr(m, "name", None)
                        or tool_name
                    )
                    call_id = str(call_id) if call_id is not None else "tool"
                    self.ui.receive_tool_reply(call_id, tool_name, m.content)
                elif (
                    m.__class__.__name__ == "AIMessage"
                    and m.content
                    and len(m.content) > 0
                ):
                    print(m.content)
                    self.ui.receive_message(m.content, "bot")

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
        self._seen_msgs = 0
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
