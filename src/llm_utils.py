"""
LLM Utilities - Multi-Provider Language Model Management
=======================================================

This module provides a unified interface for working with multiple LLM providers
(OpenAI, Anthropic, Google, DeepSeek) in the Data Science Agent system.

Core Features:
-------------
1. **Provider Abstraction**: Single interface for all LLM providers
2. **Credential Management**: Loads API keys from credentials.txt
3. **Model Configuration**: Pre-configured models with descriptions
4. **Custom Model Support**: Create LLMs with specific settings per agent
5. **Streaming & Callbacks**: Built-in support for real-time token streaming

Supported Providers:
-------------------
- **OpenAI**: GPT-4, GPT-4o, o3 reasoning models
- **Anthropic**: Claude Opus, Sonnet, Haiku models
- **Google Gemini**: Gemini 2.5 Pro and Flash
- **DeepSeek**: Chat and Reasoner models

Model Configuration Structure:
-----------------------------
Each provider has a configuration with:
- Display name for UI
- Available models with API names and descriptions
- Default temperature settings
- Recommended models per agent type

Key Functions:
-------------
- **load_credentials()**: Finds and loads API keys from credentials.txt
- **get_llm_provider()**: Creates LLM instance for specific provider
- **create_custom_llm()**: Creates LLM with custom settings and logging
- **get_recommended_model()**: Returns best model for agent type

Agent Recommendations:
---------------------
- **Orchestrator**: Needs strong reasoning (GPT-4, Claude Sonnet)
- **Research**: Fast search generation (GPT-4o-mini, Claude Haiku)
- **Coding**: Best code generation (GPT-4, Claude Sonnet)

Credential File Format:
----------------------
```
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-xxx
DEEPSEEK_API_KEY=xxx
GEMINI_API_KEY=xxx
```

Interview Notes:
---------------
- This is the CENTRAL LLM management module - all agents use this
- Supports both environment variables and file-based credentials
- The model names are mapped (e.g., "claude-sonnet-4" → actual API name)
- Includes automatic token usage logging via callbacks
- Falls back gracefully when specific models aren't available
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from work_dir_manager import log_llm_interaction
from langchain_deepseek import ChatDeepSeek


def load_credentials(credentials_file: str = None) -> Dict[str, str]:
    """
    Load API credentials from a file.
    
    Args:
        credentials_file: Path to credentials file. If None, looks for:
                         1. ../credentials.txt (parent directory)
                         2. ./credentials.txt (current directory)
                         
    Returns:
        Dictionary of API keys
    """
    if credentials_file is None:
        # Try to find credentials file
        possible_paths = [
            Path(__file__).parent.parent.parent / "credentials.txt",  # ../open_deep_research/credentials.txt
            Path(__file__).parent.parent / "credentials.txt",         # ./credentials.txt
            Path("credentials.txt"),                                   # current directory
        ]
        
        for path in possible_paths:
            if path.exists():
                credentials_file = str(path)
                print(f"Found credentials file at: {credentials_file}")
                break
        else:
            print("Warning: No credentials.txt file found")
            return {}
    
    credentials = {}
    
    try:
        with open(credentials_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    credentials[key.strip()] = value.strip()
                    # Also set as environment variable
                    os.environ[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error loading credentials: {e}")
    
    return credentials


def get_llm_provider(
    provider: Literal["openai", "anthropic", "gemini", "deepseek"] = "openai",
    model: Optional[str] = None,
    temperature: float = 0.0,
    credentials: Optional[Dict[str, str]] = None,
    streaming: bool = False,
    callbacks: Optional[list] = None,
    **kwargs
) -> Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ChatDeepSeek]:
    """
    Get an LLM instance based on the provider.
    
    Args:
        provider: LLM provider to use
        model: Model name (if None, uses default for provider)
        temperature: Temperature for generation
        credentials: API credentials dict (if None, loads from file)
        **kwargs: Additional arguments for the LLM
        
    Returns:
        LLM instance
    """
    # Load credentials if not provided
    if credentials is None:
        credentials = load_credentials()
    
    # Provider-specific configuration
    if provider == "openai":
        api_key = credentials.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in credentials or environment")
        
        model = model or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            streaming=streaming,
            callbacks=callbacks,
            **kwargs
        )
    
    elif provider == "anthropic":
        api_key = credentials.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in credentials or environment")
        
        model = model or "claude-3-haiku-20240307"
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=api_key,
            streaming=streaming,
            stream_usage=True,  # Enable token usage tracking in streaming
            callbacks=callbacks,
            **kwargs
        )
    
    elif provider == "gemini":
        api_key = credentials.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Try Google API key as alternative
            api_key = credentials.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in credentials or environment")
        
        model = model or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            streaming=streaming,
            callbacks=callbacks,
            **kwargs
        )
    
    elif provider == "deepseek":
        api_key = credentials.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in credentials or environment")
        
        model = model or "deepseek-chat"
        return ChatDeepSeek(
            model=model,
            temperature=temperature,
            api_key=api_key,
            streaming=streaming,
            callbacks=callbacks,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Configuration defaults
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", None)


def get_default_llm(**kwargs) -> Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ChatDeepSeek]:
    """
    Get the default LLM instance based on environment configuration.
    
    Uses LLM_PROVIDER and LLM_MODEL environment variables if set.
    
    Args:
        **kwargs: Additional arguments for the LLM (including streaming and callbacks)
        
    Returns:
        LLM instance
    """
    # Extract streaming and callbacks from kwargs if present
    streaming = kwargs.pop('streaming', False)
    callbacks = kwargs.pop('callbacks', None)
    
    return get_llm_provider(
        provider=DEFAULT_LLM_PROVIDER,
        model=DEFAULT_LLM_MODEL,
        streaming=streaming,
        callbacks=callbacks,
        **kwargs
    )


# Auto-load credentials on import
CREDENTIALS = load_credentials()


# Model configurations for each provider
MODEL_CONFIGS = {
    "openai": {
        "display_name": "OpenAI",
        "models": {
            "gpt-4.1": {
                "model_name": "gpt-4.1",
                "description": "Latest GPT-4.1 (Quality)",
                "temperature_default": 0.0
            },
            "gpt-4.1-mini": {
                "model_name": "gpt-4.1-mini", 
                "description": "GPT-4.1 Mini (Balanced)",
                "temperature_default": 0.0
            },
            "gpt-4.1-nano": {
                "model_name": "gpt-4.1-nano",
                "description": "GPT-4.1 Nano (Fast)",
                "temperature_default": 0.0
            },
            "gpt-4o": {
                "model_name": "gpt-4o",
                "description": "GPT-4o (Quality)",
                "temperature_default": 0.0
            },
            "gpt-4o-mini": {
                "model_name": "gpt-4o-mini",
                "description": "GPT-4o Mini (Balanced)",
                "temperature_default": 0.0
            },
            "o3": {
                "model_name": "o3",
                "description": "O3 Reasoning Model (Quality)",
                "temperature_default": 0.0
            },
            "o4-mini": {
                "model_name": "o4-mini",
                "description": "O4 Mini (Fast Reasoning)",
                "temperature_default": 0.0
            }
        },
        "default": "gpt-4.1"
    },
    "anthropic": {
        "display_name": "Anthropic",
        "models": {
            "claude-opus-4": {
                "model_name": "claude-3-opus-20240229",  # Map to actual API name
                "description": "Claude Opus 4 (Quality)",
                "temperature_default": 0.0
            },
            "claude-sonnet-4": {
                "model_name": "claude-sonnet-4-20250514",  # Map to actual API name
                "description": "Claude Sonnet 4 (Balanced)", 
                "temperature_default": 0.0
            },
            "claude-haiku-3.5": {
                "model_name": "claude-3-5-haiku-20241022",
                "description": "Claude 3.5 Haiku (Fast)",
                "temperature_default": 0.0
            }
        },
        "default": "claude-sonnet-4"
    },
    "gemini": {
        "display_name": "Google Gemini",
        "models": {
            "gemini-2.5-pro": {
                "model_name": "gemini-2.5-pro-preview-06-05",  # Use correct API name
                "description": "Gemini 2.5 Pro (Quality)",
                "temperature_default": 0.0
            },
            "gemini-2.5-flash": {
                "model_name": "gemini-2.5-flash-preview-05-20",  # Use correct API name
                "description": "Gemini 2.5 Flash (Fast)",
                "temperature_default": 0.0
            }
        },
        "default": "gemini-2.5"
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "models": {
            "deepseek-chat": {
                "model_name": "deepseek-chat",
                "description": "DeepSeek Chat (Balanced)",
                "temperature_default": 0.0
            },
            "deepseek-reasoner": {
                "model_name": "deepseek-reasoner",
                "description": "DeepSeek Reasoner (Quality)",
                "temperature_default": 0.0
            }
        },
        "default": "deepseek-chat"
    }
}

# Agent-specific model recommendations
AGENT_MODEL_RECOMMENDATIONS = {
    "orchestrator": {
        "recommended": {
            "openai": "gpt-4.1",
            "anthropic": "claude-sonnet-4-20250514", 
            "gemini": "gemini-2.5-pro",
            "deepseek": "deepseek-reasoner"
        },
        "description": "Strategic planning and coordination"
    },
    "research": {
        "recommended": {
            "openai": "gpt-4.1",
            "anthropic": "claude-sonnet-4",
            "gemini": "gemini-2.5-pro", 
            "deepseek": "deepseek-chat"
        },
        "description": "Information retrieval and search"
    },
    "coding": {
        "recommended": {
            "openai": "gpt-4.1",
            "anthropic": "claude-sonnet-4",
            "gemini": "gemini-2.5-pro",
            "deepseek": "deepseek-chat"
        },
        "description": "Code generation and analysis"
    }
}

def get_available_providers():
    """Get list of available LLM providers."""
    return list(MODEL_CONFIGS.keys())

def get_models_for_provider(provider: str):
    """Get available models for a specific provider."""
    if provider in MODEL_CONFIGS:
        return MODEL_CONFIGS[provider]["models"]
    return {}

def get_model_display_options(provider: str):
    """Get model options formatted for Streamlit dropdown."""
    models = get_models_for_provider(provider)
    return {
        model_id: f"{model_id} - {config['description']}"
        for model_id, config in models.items()
    }

def get_recommended_model(agent_type: str, provider: str):
    """Get recommended model for specific agent and provider."""
    if agent_type in AGENT_MODEL_RECOMMENDATIONS:
        return AGENT_MODEL_RECOMMENDATIONS[agent_type]["recommended"].get(provider)
    return MODEL_CONFIGS.get(provider, {}).get("default")

def create_custom_llm(
    provider: str,
    model_id: str, 
    temperature: float = None,
    streaming: bool = True,
    callbacks: List = None,
    agent_name: str = "unknown",
    max_tokens: int = None,
):
    """
    Create a custom LLM instance with specified provider and model.
    
    Args:
        provider: LLM provider (openai, anthropic, gemini, deepseek)
        model_id: Model identifier
        temperature: Temperature override
        streaming: Enable streaming
        callbacks: Custom callbacks
        agent_name: Agent name for logging
        
    Returns:
        Configured LLM instance
    """
    if provider not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[provider]["models"].get(model_id)
    if not model_config:
        raise ValueError(f"Unsupported model: {model_id} for provider: {provider}")
    
    # Use temperature from config if not specified
    if temperature is None:
        temperature = model_config["temperature_default"]
    
    # Get actual model name for API
    actual_model_name = model_config["model_name"]
    
    # Add logging callback
    if callbacks is None:
        callbacks = []
    
    from llm_logging_wrapper import LLMLoggingCallback
    logging_callback = LLMLoggingCallback(agent_name)
    callbacks = [logging_callback] + (callbacks or [])
    
    # Create LLM based on provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        # Get API key from credentials or environment
        api_key = CREDENTIALS.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in credentials or environment")
        
        return ChatOpenAI(
            model_name=actual_model_name,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Get API key from credentials or environment
        api_key = CREDENTIALS.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in credentials or environment")
        
        return ChatAnthropic(
            model=actual_model_name,
            temperature=temperature,
            streaming=streaming,
            stream_usage=True,  # Enable token usage tracking in streaming
            callbacks=callbacks,
            anthropic_api_key=api_key,
            max_tokens=max_tokens,
        )
    
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Get API key from credentials or environment
        api_key = CREDENTIALS.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or CREDENTIALS.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in credentials or environment")
        
        return ChatGoogleGenerativeAI(
            model=actual_model_name,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
            google_api_key=api_key,
            max_tokens=max_tokens,
        )
    
    elif provider == "deepseek":
        # DeepSeek uses OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        # Get API key from credentials or environment
        api_key = CREDENTIALS.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in credentials or environment")
        
        return ChatOpenAI(
            model_name=actual_model_name,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=api_key,
            max_tokens=max_tokens,
        )
    
    else:
        raise ValueError(f"Provider {provider} not implemented") 