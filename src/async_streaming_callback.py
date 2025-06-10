"""
Async-safe streaming callback handler for collecting agent output.
"""

from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from datetime import datetime
import json
from dataclasses import dataclass, field
from threading import Lock
from llm_logging_wrapper import LLMLoggingCallback


@dataclass
class AgentMessage:
    """Represents a message from an agent."""
    timestamp: str
    agent_name: str
    message_type: str  # 'thinking', 'token', 'complete', 'error', 'tool_start', 'tool_end'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncSafeStreamingHandler(BaseCallbackHandler):
    """Thread-safe callback handler that collects messages without accessing UI."""
    
    def __init__(self, agent_name: str = "Agent"):
        """Initialize the async-safe streaming handler."""
        self.agent_name = agent_name
        self.messages: List[AgentMessage] = []
        self.current_tokens: List[str] = []
        self.current_text = ""
        self._lock = Lock()
        
    def add_message(self, message_type: str, content: str, metadata: Dict[str, Any] = None):
        """Thread-safe method to add a message."""
        with self._lock:
            msg = AgentMessage(
                timestamp=datetime.now().strftime('%H:%M:%S'),
                agent_name=self.agent_name,
                message_type=message_type,
                content=content,
                metadata=metadata or {}
            )
            self.messages.append(msg)
    
    def get_messages(self) -> List[AgentMessage]:
        """Get a copy of all messages."""
        with self._lock:
            return self.messages.copy()
    
    def clear_messages(self):
        """Clear all messages."""
        with self._lock:
            self.messages.clear()
            
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.current_tokens = []
        self.current_text = ""
        self.add_message("thinking", "Starting to think...")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token."""
        self.current_tokens.append(token)
        self.current_text += token
        # Don't add every token as a message to avoid overwhelming
        # Instead, we'll get the full text at the end
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if self.current_text:
            self.add_message("complete", self.current_text)
        else:
            # Extract text from response if no streaming
            try:
                text = response.generations[0][0].text
                self.add_message("complete", text)
            except:
                self.add_message("complete", "Response completed")
    
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.add_message("error", f"Error: {str(error)}")
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        tool_name = serialized.get("name", "Unknown Tool")
        self.add_message("tool_start", f"Using tool: {tool_name}", {"input": input_str})
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        # Truncate long outputs
        display_output = output[:500] + "..." if len(output) > 500 else output
        self.add_message("tool_end", f"Tool result: {display_output}")
    
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        self.add_message("error", f"Tool error: {str(error)}")


class MultiAgentAsyncHandler:
    """Manages async-safe handlers for multiple agents."""
    
    def __init__(self):
        """Initialize multi-agent handler."""
        self.agent_handlers: Dict[str, AsyncSafeStreamingHandler] = {}
        self._lock = Lock()
        
    def get_handler(self, agent_name: str) -> AsyncSafeStreamingHandler:
        """Get or create a handler for a specific agent."""
        with self._lock:
            if agent_name not in self.agent_handlers:
                self.agent_handlers[agent_name] = AsyncSafeStreamingHandler(agent_name)
            return self.agent_handlers[agent_name]
    
    def create_agent_callbacks(self, agent_name: str) -> List[BaseCallbackHandler]:
        """Create callbacks list for an agent with both streaming and logging."""
        callbacks = [
            self.get_handler(agent_name),  # Streaming handler
            LLMLoggingCallback(agent_name)  # LLM logging handler
        ]
        return callbacks
    
    def get_all_messages(self) -> Dict[str, List[AgentMessage]]:
        """Get all messages from all agents."""
        with self._lock:
            return {
                agent_name: handler.get_messages()
                for agent_name, handler in self.agent_handlers.items()
            }
    
    def format_messages_for_display(self) -> List[str]:
        """Format all messages for display."""
        all_messages = []
        for agent_name, messages in self.get_all_messages().items():
            for msg in messages:
                if msg.message_type == "thinking":
                    all_messages.append(f"[{msg.timestamp}] 🧠 **{msg.agent_name}** - {msg.content}")
                elif msg.message_type == "complete":
                    # Truncate long responses
                    content = msg.content[:1000] + "..." if len(msg.content) > 1000 else msg.content
                    all_messages.append(f"[{msg.timestamp}] ✅ **{msg.agent_name}** - Completed:\n{content}")
                elif msg.message_type == "error":
                    all_messages.append(f"[{msg.timestamp}] ❌ **{msg.agent_name}** - {msg.content}")
                elif msg.message_type == "tool_start":
                    all_messages.append(f"[{msg.timestamp}] 🔧 **{msg.agent_name}** - {msg.content}")
                elif msg.message_type == "tool_end":
                    all_messages.append(f"[{msg.timestamp}] 📊 **{msg.agent_name}** - {msg.content}")
        return all_messages 