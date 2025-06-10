#!/usr/bin/env python3
"""
Run script for the Data Science Agent.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def check_environment():
    """Check if required environment variables are set."""
    # Import here to trigger credential loading
    try:
        from llm_utils import CREDENTIALS
        print("✅ Loaded credentials from file")
    except Exception as e:
        print(f"⚠️  Could not load credentials: {e}")
    
    # Check for at least one LLM API key
    llm_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"]
    has_llm_key = any(os.getenv(key) for key in llm_keys)
    
    if not has_llm_key:
        print("❌ No LLM API key found. You need at least one of:")
        for key in llm_keys:
            print(f"   - {key}")
        print("\nSet in credentials.txt or environment variables")
        return False
    
    # Check for Riza key (optional now with local execution)
    if not os.getenv("RIZA_API_KEY"):
        print("⚠️  No RIZA_API_KEY found - will use local code execution")
        print("   (Remote execution requires RIZA_API_KEY)")
    
    # Show which providers are available
    print("\n📋 Available LLM providers:")
    for key, provider in [
        ("OPENAI_API_KEY", "OpenAI"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("GEMINI_API_KEY", "Google Gemini"),
        ("DEEPSEEK_API_KEY", "DeepSeek")
    ]:
        if os.getenv(key):
            print(f"   ✅ {provider}")
    
    return True


def setup_logging(debug=False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Set specific loggers
    if debug:
        # Enable debug logging for our modules
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("agents").setLevel(logging.DEBUG)
        logging.getLogger("tools").setLevel(logging.DEBUG)
        
        # Disable noisy third-party loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)
        
        print("🐛 DEBUG MODE ENABLED - Verbose logging activated")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Science Agent")
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Use new multi-agent interface with per-agent model selection"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve code execution without user confirmation (use with caution!)"
    )
    
    # Model selection arguments for each agent
    parser.add_argument(
        "--orchestrator-provider",
        choices=["openai", "anthropic", "google", "deepseek"],
        help="LLM provider for orchestrator agent"
    )
    parser.add_argument(
        "--orchestrator-model",
        help="Model for orchestrator agent (e.g., gpt-4o, claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--research-provider", 
        choices=["openai", "anthropic", "google", "deepseek"],
        help="LLM provider for research agent"
    )
    parser.add_argument(
        "--research-model",
        help="Model for research agent (e.g., gpt-4o-mini, claude-3-5-haiku-20241022)"
    )
    parser.add_argument(
        "--coding-provider",
        choices=["openai", "anthropic", "google", "deepseek"], 
        help="LLM provider for coding agent"
    )
    parser.add_argument(
        "--coding-model",
        help="Model for coding agent (e.g., gpt-4o, gemini-2.5-pro-preview-06-05)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(debug=args.debug)
    
    print("🔬 Data Science Agent")
    print("=" * 40)
    
    if args.debug:
        print("🐛 Running in DEBUG mode")
        # Set environment variable for other modules to check
        os.environ["DEBUG_MODE"] = "1"
    
    if args.auto_approve:
        print("⚠️  AUTO-APPROVE MODE: Code will execute without user confirmation!")
        print("   Use Ctrl+C to stop if needed")
        os.environ["AUTO_APPROVE_CODE"] = "1"
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Set model configurations if provided
    if any([args.orchestrator_provider, args.research_provider, args.coding_provider]):
        print("\n🤖 Custom model configuration detected:")
        
        if args.orchestrator_provider:
            print(f"   🧠 Orchestrator: {args.orchestrator_provider}")
            if args.orchestrator_model:
                print(f"      Model: {args.orchestrator_model}")
        
        if args.research_provider:
            print(f"   🔍 Research: {args.research_provider}")
            if args.research_model:
                print(f"      Model: {args.research_model}")
        
        if args.coding_provider:
            print(f"   💻 Coding: {args.coding_provider}")
            if args.coding_model:
                print(f"      Model: {args.coding_model}")
        
        # Set environment variables for the agents to pick up
        if args.orchestrator_provider:
            os.environ["ORCHESTRATOR_PROVIDER"] = args.orchestrator_provider
        if args.orchestrator_model:
            os.environ["ORCHESTRATOR_MODEL"] = args.orchestrator_model
        if args.research_provider:
            os.environ["RESEARCH_PROVIDER"] = args.research_provider
        if args.research_model:
            os.environ["RESEARCH_MODEL"] = args.research_model
        if args.coding_provider:
            os.environ["CODING_PROVIDER"] = args.coding_provider
        if args.coding_model:
            os.environ["CODING_MODEL"] = args.coding_model
    
    # Note: Vector database loading is handled by the vector_search tool
    # It will automatically find embeddings in various locations:
    # - openai_kaggle_db/
    # - open_data_science_agent/data/vector_db/{model_name}/
    # No need to check here as the tool handles it gracefully
    
    # Choose which app to run
    if args.multi_agent or any([args.orchestrator_provider, args.research_provider, args.coding_provider]):
        # Use new multi-agent app
        print("\n🚀 Starting MULTI-AGENT Streamlit app...")
        print("   Features: Per-agent model selection, enhanced debugging")
        print("   Open http://localhost:8501 in your browser\n")
        
        app_path = Path(__file__).parent / "app" / "streamlit_app.py"
        
        # Set PYTHONPATH to include src directory  
        env = os.environ.copy()
        src_path_str = str(src_path)
        env['PYTHONPATH'] = src_path_str + os.pathsep + env.get('PYTHONPATH', '')
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], env=env)
    else:
        # Use legacy single-agent app
        print("\n🚀 Starting LEGACY Streamlit app...")
        print("   Note: Use --multi-agent for new features")
        print("   Open http://localhost:8501 in your browser\n")
        
        app_path = src_path / "app.py"
        
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env['PYTHONPATH'] = str(src_path) + os.pathsep + env.get('PYTHONPATH', '')
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], env=env)


if __name__ == "__main__":
    main() 