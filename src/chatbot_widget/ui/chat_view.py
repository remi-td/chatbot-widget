import ast
import datetime as _dt
import html
import json
import uuid
import warnings

import ipywidgets as widgets
from IPython.display import HTML, display

from .components.chat_bubble import ChatBubble
from .components.input_bar import InputBar
from .components.renderers import collapsible
from .components.scroll_box import ScrollBox


class ChatView:
    """Main chat UI (publisher/subscriber pattern)."""

    def __init__(self):
        # Core UI components
        self.chat_box = ScrollBox()
        self.input_bar = InputBar()

        # Observer callbacks
        self._on_send_callback = None

        # Streaming helpers
        self._active_streams: dict[str, ChatBubble] = {}
        self._thinking_widget: widgets.HTML | None = None
        self._tool_containers: dict[str, widgets.VBox] = {}
        self._tool_metadata: dict[str, str] = {}

        # Layout
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            container_layout = widgets.Layout(
                align_items="center",
                width="100%",
                max_width="900px",
                height="auto",
                overflow="visible",
                margin="0 auto",
                padding="20px",
                border="none",
                flex_flow="column",
                background="#c3c5c9",
                border_radius="12px",
            )

        self.container = widgets.VBox(
            [self.chat_box, self.input_bar.widget],
            layout=container_layout,
        )

        # Welcome message
        welcome = widgets.HTML(
            "<div style='text-align:center;color:#555;padding:12px;'>"
            "💬 <b>Welcome! How can I help you today?</b></div>"
        )
        self.chat_box.children = (welcome,)

        # Hook input button to handler
        self.input_bar.button.on_click(self._handle_send)

        # Inject small CSS tweak to prevent notebook scroll
        display(
            HTML(
                """
        <style>
        .jp-OutputArea-output, .jp-OutputArea-child, .widget-html-content {
            overflow: visible !important;
            max-height: none !important;
        }

        .cbw-spinner-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin: 0 2px;
            background: #6c63ff;
            animation: cbw-pulse 0.9s infinite ease-in-out;
            display: inline-block;
        }
        .cbw-spinner-dot:nth-child(2) { animation-delay: 0.15s; }
        .cbw-spinner-dot:nth-child(3) { animation-delay: 0.3s; }

        @keyframes cbw-pulse {
            0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
            40% { opacity: 1; transform: scale(1); }
        }
        </style>
        """
            )
        )

    # ------------------------------------------------------------------ #
    # Observer registration
    # ------------------------------------------------------------------ #
    def on_send(self, callback):
        """Register callback(msg: str) when user sends a message."""
        self._on_send_callback = callback

    # ------------------------------------------------------------------ #
    # Internal event handling
    # ------------------------------------------------------------------ #
    def _handle_send(self, _=None):
        """Triggered when user clicks send."""
        text = self.input_bar.input.value.strip()
        if not text:
            return

        self.input_bar.clear()
        self.chat_box.children += (ChatBubble(text, sender="user").widget,)

        if self._on_send_callback:
            self._on_send_callback(text)

    # ------------------------------------------------------------------ #
    # Public display / response methods
    # ------------------------------------------------------------------ #
    def set_busy(self, busy: bool):
        """Toggle busy state for the input bar."""
        self.input_bar.set_busy(busy)
        if not busy:
            self.hide_waiting_indicator()
            self._active_streams.clear()
        else:
            # prepare to track fresh tool activity for this run
            self._tool_containers = {
                k: v for k, v in self._tool_containers.items() if v in self.chat_box.children
            }
            self._tool_metadata = {
                k: self._tool_metadata.get(k, "tool") for k in self._tool_containers.keys()
            }

    def show_waiting_indicator(self, message: str = ""):
        """Display animated indicator while waiting for the model response."""
        if self._thinking_widget is not None:
            return
        spinner = widgets.HTML(
            value=(
                "<div style='display:flex;align-items:center;gap:8px;margin:4px 0;color:#3d3d63;"
                "font-size:13px;background:#f1f3ff;border-radius:12px;padding:8px 12px;'>"
                "<span class='cbw-spinner-dot'></span>"
                "<span class='cbw-spinner-dot'></span>"
                "<span class='cbw-spinner-dot'></span>"
                f"<span style='margin-left:6px;'>{message}</span>"
                "</div>"
            )
        )
        self.chat_box.children += (spinner,)
        self._thinking_widget = spinner

    def hide_waiting_indicator(self):
        """Remove the waiting indicator widget if present."""
        if self._thinking_widget is None:
            return
        children = list(self.chat_box.children)
        if self._thinking_widget in children:
            children.remove(self._thinking_widget)
            self.chat_box.children = tuple(children)
        self._thinking_widget = None

    def receive_message(self, text, sender: str = "bot"):
        """Display a message (from controller)."""
        if not isinstance(text, str):
            text = str(text)
        if sender != "user":
            self.hide_waiting_indicator()
        bubble = ChatBubble(text, sender)
        self.chat_box.children += (bubble.widget,)

    def start_stream(self, sender: str = "bot", initial_text: str = "") -> str:
        """Create a bubble that can be updated incrementally."""
        message_id = str(uuid.uuid4())
        bubble = ChatBubble(initial_text, sender)
        self._active_streams[message_id] = bubble
        self.chat_box.children += (bubble.widget,)
        return message_id

    def stream_update(self, message_id: str, text: str):
        """Update the text of an active streaming bubble."""
        bubble = self._active_streams.get(message_id)
        if bubble:
            bubble.update_text(text)

    def end_stream(self, message_id: str, final_text: str | None = None):
        """Finalize a streaming bubble (optionally overriding the final text)."""
        bubble = self._active_streams.pop(message_id, None)
        if bubble:
            if final_text is not None:
                bubble.update_text(final_text)
            children = list(self.chat_box.children)
            if bubble.widget in children:
                children.remove(bubble.widget)
                children.append(bubble.widget)
                self.chat_box.children = tuple(children)
        if not self._active_streams:
            self.hide_waiting_indicator()

    def receive_tool_call(self, call_id: str, tool_name: str, tool_args):
        """Render a tool call inline with a timestamp."""
        timestamp = _dt.datetime.now().strftime("%H:%M:%S")
        if not isinstance(tool_args, str):
            tool_args = str(tool_args)
        header = self._format_tool_header("call", call_id, tool_name, timestamp)
        call_widget = collapsible(header, self._format_tool_payload(tool_args))

        container = self._tool_containers.get(call_id)
        if container:
            children = list(container.children)
            if children:
                children[0] = call_widget
            else:
                children.append(call_widget)
            container.children = tuple(children)
        else:
            container = widgets.VBox([call_widget], layout=widgets.Layout(width="100%", margin="4px 0"))
            self._tool_containers[call_id] = container
            self.chat_box.children += (container,)
        self._tool_metadata[call_id] = tool_name

    def receive_tool_reply(self, call_id: str, tool_name: str | None, tool_reply):
        """Render a tool reply inline with a timestamp, beneath its matching call."""
        timestamp = _dt.datetime.now().strftime("%H:%M:%S")
        if not isinstance(tool_reply, str):
            tool_reply = str(tool_reply)
        label = tool_name or self._tool_metadata.get(call_id, "tool")
        header = self._format_tool_header("result", call_id, label, timestamp)
        reply_widget = collapsible(header, self._format_tool_payload(tool_reply))

        container = self._tool_containers.get(call_id)
        if container:
            children = list(container.children)
            if len(children) == 0:
                children.append(reply_widget)
            elif len(children) == 1:
                children.append(reply_widget)
            else:
                children[1] = reply_widget
            container.children = tuple(children)
        else:
            container = widgets.VBox([reply_widget], layout=widgets.Layout(width="100%", margin="4px 0"))
            self._tool_containers[call_id] = container
            self.chat_box.children += (container,)
        self._tool_metadata.setdefault(call_id, label)

    def display(self):
        """Render chat UI in the notebook."""
        display(self.container)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_tool_payload(payload: str) -> str:
        text = payload if isinstance(payload, str) else str(payload)
        stripped = text.strip()
        parsed = None
        try:
            parsed = json.loads(stripped)
        except Exception:
            pass
        if parsed is None:
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                pass
        if parsed is not None:
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            # Unescape escape sequences inside JSON string values so they
            # render as actual whitespace in the <pre> block, not literal \n \t
            formatted = formatted.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '')
        else:
            formatted = stripped
        escaped = html.escape(formatted)
        return f"<pre style='margin:0;white-space:pre-wrap;'>{escaped}</pre>"

    def _format_tool_header(self, kind: str, call_id: str, tool_name: str, timestamp: str) -> str:
        short_id = html.escape(call_id[:8])
        full_id = html.escape(call_id)
        safe_name = html.escape(tool_name)
        safe_timestamp = html.escape(timestamp)
        if kind == "call":
            return (
                f"🔧 Tool call · {safe_name} "
                f"<span title='{full_id}'>[{short_id}]</span> @ {safe_timestamp}"
            )
        return (
            f"📦 Tool result "
            f"<span title='{full_id}'>[{short_id}]</span> @ {safe_timestamp}"
        )