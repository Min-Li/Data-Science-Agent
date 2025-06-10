"""
Streaming callback handler for real-time token display.
"""

from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
try:
    from langchain_core.agents import AgentAction, AgentFinish
except ImportError:
    # Fallback for older versions
    AgentAction = None
    AgentFinish = None
import streamlit as st
from datetime import datetime
import json


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming tokens to Streamlit UI."""
    
    def __init__(self, container=None, agent_name: str = "Agent"):
        """
        Initialize the streaming callback handler.
        
        Args:
            container: Streamlit container for output
            agent_name: Name of the agent for display
        """
        self.container = container
        self.agent_name = agent_name
        self.current_tokens = []
        self.current_text = ""
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        if self.container:
            with self.container:
                st.markdown(f"**[{datetime.now().strftime('%H:%M:%S')}] {self.agent_name}** 🤔 Thinking...")
        self.current_tokens = []
        self.current_text = ""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.current_tokens.append(token)
        self.current_text += token
        
        if self.container:
            # Use session state to track placeholders
            if not hasattr(st.session_state, 'stream_placeholders'):
                st.session_state.stream_placeholders = {}
            
            key = f"{self.agent_name}_current"
            
            with self.container:
                # Get or create placeholder
                if key not in st.session_state.stream_placeholders:
                    st.session_state.stream_placeholders[key] = st.empty()
                
                # Update the placeholder
                st.session_state.stream_placeholders[key].markdown(
                    f"**[{datetime.now().strftime('%H:%M:%S')}] {self.agent_name}**\n\n{self.current_text}",
                    unsafe_allow_html=True
                )
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if self.container:
            # Clear the current placeholder
            key = f"{self.agent_name}_current"
            if hasattr(st.session_state, 'stream_placeholders') and key in st.session_state.stream_placeholders:
                st.session_state.stream_placeholders[key].empty()
                del st.session_state.stream_placeholders[key]
            
            with self.container:
                st.markdown(f"**[{datetime.now().strftime('%H:%M:%S')}] {self.agent_name}** ✅ Complete")
                if self.current_text:
                    # Show final text in an expander
                    with st.expander(f"{self.agent_name} Response", expanded=False):
                        st.markdown(self.current_text)
                st.divider()
    
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        if self.container:
            with self.container:
                st.error(f"**{self.agent_name} Error:** {str(error)}")
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        tool_name = serialized.get("name", "Unknown Tool")
        if self.container:
            with self.container:
                st.info(f"**[{datetime.now().strftime('%H:%M:%S')}] {self.agent_name}** 🔧 Using tool: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        if self.container:
            with self.container:
                # Truncate long outputs
                display_output = output[:500] + "..." if len(output) > 500 else output
                st.success(f"**Tool Result:** {display_output}")
    
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        if self.container:
            with self.container:
                st.error(f"**Tool Error:** {str(error)}")
    
    def on_agent_action(self, action, **kwargs: Any) -> None:
        """Run on agent action."""
        if self.container and AgentAction:
            with self.container:
                st.info(f"**{self.agent_name} Action:** {action.tool} - {action.tool_input}")
    
    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        """Run on agent finish."""
        if self.container and AgentFinish:
            with self.container:
                st.success(f"**{self.agent_name} Finished**")


class MultiAgentStreamingHandler:
    """Manages streaming for multiple agents."""
    
    def __init__(self, container=None):
        """Initialize multi-agent streaming handler."""
        self.container = container
        self.agent_handlers = {}
        
    def get_handler(self, agent_name: str) -> StreamingCallbackHandler:
        """Get or create a handler for a specific agent."""
        if agent_name not in self.agent_handlers:
            self.agent_handlers[agent_name] = StreamingCallbackHandler(
                container=self.container,
                agent_name=agent_name
            )
        return self.agent_handlers[agent_name]
    
    def create_agent_callbacks(self, agent_name: str) -> List[BaseCallbackHandler]:
        """Create callbacks list for an agent."""
        return [self.get_handler(agent_name)] 