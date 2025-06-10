"""
LLM Logging Wrapper for tracking all interactions with language models.

This wrapper captures:
- All messages sent to LLMs
- All responses received
- Token usage information
- Timing information
"""

import time
import json
from typing import Dict, Any, List, Optional, Union
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from work_dir_manager import log_llm_interaction, log_message


class LLMLoggingCallback(BaseCallbackHandler):
    """Callback handler to log LLM interactions."""
    
    def __init__(self, agent_name: str = "unknown"):
        self.agent_name = agent_name
        self.start_time = None
        self.current_request = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        self.start_time = time.time()
        self.current_request = {
            "prompts": prompts,
            "serialized": serialized,
            "kwargs": kwargs
        }
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running."""
        if self.current_request and self.start_time:
            duration = time.time() - self.start_time
            
            # Extract token usage if available
            input_tokens = 0
            output_tokens = 0
            
            # First try to get token usage from llm_output (OpenAI style)
            if response.llm_output and isinstance(response.llm_output, dict):
                token_usage = response.llm_output.get('token_usage', {})
                if isinstance(token_usage, dict):
                    input_tokens = token_usage.get('prompt_tokens', 0)
                    output_tokens = token_usage.get('completion_tokens', 0)
                
                # Also check 'usage' key (Anthropic style in llm_output)
                if input_tokens == 0 and output_tokens == 0:
                    usage = response.llm_output.get('usage', {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
            
            # If still no tokens, check if there's usage_metadata in the generations
            # This handles Anthropic models with stream_usage=True
            if input_tokens == 0 and output_tokens == 0 and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
                            usage_meta = gen.message.usage_metadata
                            if usage_meta:
                                input_tokens = usage_meta.get('input_tokens', 0)
                                output_tokens = usage_meta.get('output_tokens', 0)
                                break
                    if input_tokens > 0 or output_tokens > 0:
                        break
            
            # Log the interaction
            try:
                # Safely convert generations to dict format
                generations_data = []
                for gen_list in response.generations:
                    gen_data = []
                    for gen in gen_list:
                        if hasattr(gen, 'dict'):
                            gen_data.append(gen.dict())
                        else:
                            gen_data.append(str(gen))
                    generations_data.append(gen_data)
            except Exception as e:
                generations_data = [str(response.generations)]
            
            # Safely serialize llm_output to avoid JSON serialization errors
            try:
                llm_output_serializable = json.loads(json.dumps(response.llm_output, default=str))
            except:
                llm_output_serializable = str(response.llm_output)
            
            log_llm_interaction(
                agent_name=self.agent_name,
                request=self.current_request,
                response={
                    "generations": generations_data,
                    "llm_output": llm_output_serializable,
                    "run": str(response.run) if response.run else None,
                    "duration_seconds": duration
                },
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            log_message(f"🤖 {self.agent_name}: {input_tokens} input + {output_tokens} output tokens ({duration:.1f}s)")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM errors."""
        if self.current_request:
            log_llm_interaction(
                agent_name=self.agent_name,
                request=self.current_request,
                response={"error": str(error)},
                input_tokens=0,
                output_tokens=0
            )
            log_message(f"❌ {self.agent_name}: LLM error - {error}", "error")


class LLMLoggingWrapper:
    """Wrapper that adds logging to any LLM."""
    
    def __init__(self, llm: BaseLanguageModel, agent_name: str = "unknown"):
        self.llm = llm
        self.agent_name = agent_name
        self._add_logging_callback()
    
    def _add_logging_callback(self):
        """Add logging callback to the LLM."""
        logging_callback = LLMLoggingCallback(self.agent_name)
        
        # Add to existing callbacks
        if hasattr(self.llm, 'callbacks') and self.llm.callbacks:
            self.llm.callbacks.append(logging_callback)
        else:
            self.llm.callbacks = [logging_callback]
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM."""
        return getattr(self.llm, name)
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable like the original LLM."""
        return self.llm(*args, **kwargs)


def wrap_llm_with_logging(llm: BaseLanguageModel, agent_name: str = "unknown") -> BaseLanguageModel:
    """Wrap an LLM with logging capabilities."""
    return LLMLoggingWrapper(llm, agent_name)


# Convenience function to create logged callbacks
def create_logged_callbacks(agent_name: str) -> List[BaseCallbackHandler]:
    """Create a list of callbacks that includes LLM logging."""
    return [LLMLoggingCallback(agent_name)] 