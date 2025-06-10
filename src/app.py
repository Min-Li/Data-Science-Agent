"""
Streamlit app for the Data Science Agent.
"""

import streamlit as st
import asyncio
import pandas as pd
import json
import base64
from datetime import datetime
import os
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional

from graph import run_data_science_agent, solve_kaggle_problem
from state import DataScienceState
from async_streaming_callback import MultiAgentAsyncHandler
from llm_utils import get_llm_provider
from work_dir_manager import create_new_run, get_work_dir_manager, log_message


# Page configuration
st.set_page_config(
    page_title="Data Science Agent",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .code-block {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        overflow-x: auto;
    }
    .insight-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0068c9;
    }
    .visualization {
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def display_visualization(base64_img: str, idx: int):
    """Display a base64 encoded image."""
    st.markdown(
        f'<div class="visualization"><img src="data:image/png;base64,{base64_img}" style="max-width: 100%;" /></div>',
        unsafe_allow_html=True
    )


def display_kaggle_insights(insights: list):
    """Display Kaggle insights in a formatted way."""
    if not insights:
        return
    
    st.subheader("📊 Kaggle Competition Insights")
    
    for i, insight in enumerate(insights[:5]):  # Show top 5
        with st.expander(f"🏆 {insight.get('competition_name', f'Competition {i+1}')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Problem Type:** {insight.get('problem_type', 'Unknown')}")
                st.markdown(f"**Dataset:** {insight.get('dataset_description', 'N/A')}")
                
                if insight.get('key_insights'):
                    st.markdown("**Key Insights:**")
                    st.markdown(insight['key_insights'])
            
            with col2:
                if insight.get('ml_techniques'):
                    st.markdown("**Techniques Used:**")
                    for tech in insight['ml_techniques']:
                        st.markdown(f"• {tech}")
                
                if insight.get('similarity_score'):
                    st.metric("Relevance Score", f"{insight['similarity_score']:.2%}")


def display_code_with_explanation(code: str, explanation: dict):
    """Display generated code with explanation."""
    st.subheader("🐍 Generated Solution")
    
    # Display explanation
    with st.expander("📝 Solution Explanation", expanded=True):
        st.markdown(f"**Overview:** {explanation.get('overview', 'N/A')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Steps:**")
            for step in explanation.get('key_steps', []):
                st.markdown(f"• {step}")
        
        with col2:
            st.markdown("**Techniques Used:**")
            for tech in explanation.get('techniques', []):
                st.markdown(f"• {tech}")
    
    # Display code
    st.code(code, language='python')
    
    # Download button for code
    st.download_button(
        label="📥 Download Code",
        data=code,
        file_name=f"data_science_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
        mime="text/x-python"
    )


def display_execution_results(result: str, error: Optional[str], visualizations: list):
    """Display code execution results."""
    st.subheader("🚀 Execution Results")
    
    if error:
        st.error(f"**Execution Error:**\n```\n{error}\n```")
    else:
        st.success("✅ Code executed successfully!")
        
        if result:
            st.markdown("**Output:**")
            st.code(result, language='text')
        
        if visualizations:
            st.markdown("**Visualizations:**")
            cols = st.columns(min(len(visualizations), 2))
            for i, viz in enumerate(visualizations):
                with cols[i % 2]:
                    display_visualization(viz, i)


async def process_problem(problem_description: str, dataset_file, dataset_info: str, 
                         llm_provider: str = None, llm_model: str = None,
                         streaming_handler=None):
    """Process the data science problem with the agent."""
    # Create new run directory
    model_name = f"{llm_provider or 'unknown'}-{llm_model or 'default'}"
    work_dir = create_new_run(model_name)
    work_manager = get_work_dir_manager()
    
    log_message(f"🔬 Starting new data science agent run")
    log_message(f"📝 Problem: {problem_description[:100]}...")
    log_message(f"🤖 Model: {model_name}")
    
    # Save uploaded file to work directory
    temp_path = None
    if dataset_file:
        try:
            # First save to temp file, then copy to work directory
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(dataset_file.name).suffix) as tmp:
                tmp.write(dataset_file.getbuffer())
                temp_file_path = tmp.name
            
            # Copy to work directory inputs folder
            dataset_path = work_manager.copy_input_file(temp_file_path, dataset_file.name)
            temp_path = dataset_path
            log_message(f"📁 Dataset saved to work directory: {dataset_file.name}")
            
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
        except Exception as e:
            # Fallback to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(dataset_file.name).suffix) as tmp:
                tmp.write(dataset_file.getbuffer())
                temp_path = tmp.name
                log_message(f"📁 Dataset saved to temporary file: {dataset_file.name} (work dir failed: {e})")
    
    try:
        # Prepare dataset information
        dataset_dict = {
            "description": dataset_info,
            "filename": dataset_file.name if dataset_file else None
        }
        
        if temp_path:
            dataset_dict["path"] = temp_path
            
            # Try to load and get basic info
            try:
                if dataset_file.name.endswith('.csv'):
                    df = pd.read_csv(temp_path)
                    dataset_dict["shape"] = df.shape
                    dataset_dict["columns"] = df.columns.tolist()
                    dataset_dict["preview"] = df.head(10).to_dict()  # Show top 10 rows to LLM
            except:
                pass
        
        # Add work directory to dataset_dict so it can be passed to state
        dataset_dict["work_dir"] = str(work_dir)
        
        # Create callbacks if streaming is enabled
        callbacks = None
        if streaming_handler:
            # Create agent-specific callbacks
            callbacks = {
                "orchestrator": streaming_handler.create_agent_callbacks("🧠 Orchestrator"),
                "research": streaming_handler.create_agent_callbacks("🔍 Research Agent"),
                "coding": streaming_handler.create_agent_callbacks("💻 Coding Agent")
            }
        
        # Run the agent
        log_message("🤖 Starting agent workflow...")
        result = await run_data_science_agent(
            problem_description=problem_description,
            dataset_info=dataset_dict,
            max_iterations=20,
            llm_provider=llm_provider,
            llm_model=llm_model,
            callbacks=callbacks
        )
        
        # Save final results to work directory
        try:
            if result.get('generated_code'):
                work_manager.save_result("final_solution.py", result['generated_code'])
                log_message("💾 Saved final solution code")
            
            if result.get('final_summary'):
                work_manager.save_result("summary.md", result['final_summary'])
                log_message("💾 Saved final summary")
            
            # Save complete results as JSON (excluding non-serializable fields)
            # Remove messages field which contains AIMessage objects
            results_for_json = {k: v for k, v in result.items() if k != 'messages'}
            results_json = {
                "problem_description": problem_description,
                "dataset_info": dataset_dict,
                "results": results_for_json,
                "timestamp": datetime.now().isoformat()
            }
            work_manager.save_result("complete_results.json", json.dumps(results_json, indent=2))
            log_message("💾 Saved complete results")
            
            work_manager.finalize_run("completed")
            
        except Exception as e:
            log_message(f"⚠️ Error saving results: {e}", "warning")
        
        log_message("✅ Agent workflow completed")
        return result, streaming_handler
        
    finally:
        # Clean up temp file (if fallback was used)
        if temp_path and os.path.exists(temp_path) and not work_manager.work_dir:
            os.unlink(temp_path)


# Main app
def main():
    st.title("🔬 Data Science Agent")
    st.markdown("""
    This agent automatically solves data science problems by:
    1. 🔍 Searching insights from 647 Kaggle competitions
    2. 💡 Generating Python solutions based on best practices
    3. ⚡ Executing code locally (with approval) or remotely (sandbox)
    4. 📊 Producing visualizations and results
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Import credentials
        try:
            from llm_utils import CREDENTIALS, get_default_llm
            credentials_loaded = True
        except:
            credentials_loaded = False
        
        # API Keys check
        api_keys_ok = True
        
        # Check for at least one LLM key
        llm_providers = []
        for key, provider in [
            ("OPENAI_API_KEY", "openai"),
            ("ANTHROPIC_API_KEY", "anthropic"),
            ("GEMINI_API_KEY", "gemini"),
            ("DEEPSEEK_API_KEY", "deepseek")
        ]:
            if os.getenv(key):
                llm_providers.append((provider, key.replace("_API_KEY", "")))
        
        if not llm_providers:
            st.error("No LLM API keys found")
            api_keys_ok = False
        
        # Check for code execution (Riza is optional now)
        has_riza = bool(os.getenv("RIZA_API_KEY"))
        
        if api_keys_ok:
            st.success("✅ API keys configured")
            
            # Code Execution Mode selection
            st.subheader("⚡ Code Execution")
            execution_mode = st.selectbox(
                "Execution Mode",
                options=["local", "remote"],
                index=0,
                help="Local: Execute in your environment with approval\nRemote: Execute in Riza's secure sandbox"
            )
            
            if execution_mode == "local":
                st.info("🏠 **Local Mode**: Code runs in your environment with your approval before execution.")
            else:
                if has_riza:
                    st.info("☁️ **Remote Mode**: Code runs in Riza's secure sandbox.")
                else:
                    st.warning("⚠️ **Remote Mode**: Requires RIZA_API_KEY. Will fallback to local mode.")
            
            # Set environment variable for execution mode
            os.environ["CODE_EXECUTION_MODE"] = execution_mode
            
            # LLM Provider selection
            st.subheader("🤖 LLM Settings")
            provider_names = [p[0] for p in llm_providers]
            provider_labels = {
                "openai": "OpenAI",
                "anthropic": "Anthropic Claude",
                "gemini": "Google Gemini",
                "deepseek": "DeepSeek"
            }
            
            selected_provider = st.selectbox(
                "Select LLM Provider",
                provider_names,
                format_func=lambda x: provider_labels.get(x, x)
            )
            
            # Show recommended models
            from llm_utils import MODEL_CONFIGS
            if selected_provider in MODEL_CONFIGS:
                models = MODEL_CONFIGS[selected_provider]["models"]
                # Get some example models for display
                model_examples = list(models.keys())[:3]  # Show first 3 models as examples
                st.caption(f"Available models:")
                for model_id in model_examples:
                    model_desc = models[model_id]["description"]
                    st.caption(f"• {model_id}: {model_desc}")
            
            # Custom model input
            st.subheader("🔧 Model Configuration")
            
            # Get available models for the selected provider
            available_models = {}
            default_model = ""
            if selected_provider in MODEL_CONFIGS:
                available_models = MODEL_CONFIGS[selected_provider]["models"]
                default_model = MODEL_CONFIGS[selected_provider].get("default", "")
            
            if available_models:
                # Create model options with descriptions
                model_options = {}
                for model_id, config in available_models.items():
                    model_options[model_id] = f"{model_id} - {config['description']}"
                
                # Set default selection
                default_index = 0
                if default_model and default_model in model_options:
                    default_index = list(model_options.keys()).index(default_model)
                
                selected_model = st.selectbox(
                    "Select Model",
                    options=list(model_options.keys()),
                    format_func=lambda x: model_options[x],
                    index=default_index,
                    help=f"Choose a model from {MODEL_CONFIGS[selected_provider]['display_name']}"
                )
                
                # Show model details
                if selected_model in available_models:
                    model_config = available_models[selected_model]
                    st.caption(f"**Temperature:** {model_config['temperature_default']}")
                    st.caption(f"**Description:** {model_config['description']}")
            else:
                # Fallback to text input if no models configured
                selected_model = st.text_input(
                    "Model Name",
                    placeholder="Enter model name",
                    help="Enter the model name manually"
                )
            
            # Activity logging options
            st.subheader("📡 Activity Logging")
            show_streaming = st.checkbox(
                "Capture Agent Activity",
                value=True,
                help="Capture detailed logs of agent thinking and actions (displayed after completion)"
            )
        
        st.divider()
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=30,
                value=20,
                help="Maximum number of iterations for agents to refine the solution"
            )
        
        st.divider()
        
        # About section
        st.markdown("""
        ### About
        
        Built with:
        - 🦜 LangChain & LangGraph
        - ⚡ Dual Code Execution:
          - 🏠 Local (with user approval)
          - ☁️ Remote (Riza sandbox)
        - 🏆 Kaggle Competition Insights
        - 🤖 Multi-LLM Support:
          - OpenAI (GPT-3.5/4)
          - Anthropic (Claude)
          - Google (Gemini)
          - DeepSeek
        """)
    
    # Main content area
    if not api_keys_ok:
        st.warning("Please configure at least one LLM API key in your environment variables.")
        st.code("""
# At least one of these LLM keys is required:
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-google-key"
export DEEPSEEK_API_KEY="your-deepseek-key"

# Optional for remote code execution:
export RIZA_API_KEY="your-riza-key"
        """, language='bash')
        return
    
    # Input section
    st.header("📝 Problem Definition")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        problem_description = st.text_area(
            "Describe your data science problem:",
            placeholder="Example: Build a model to predict customer churn based on usage patterns and demographics.",
            height=120
        )
    
    with col2:
        dataset_file = st.file_uploader(
            "Upload dataset (optional):",
            type=['csv', 'xlsx', 'json'],
            help="Upload your dataset for analysis"
        )
        
        dataset_info = st.text_area(
            "Dataset description:",
            placeholder="Describe your dataset features, target variable, etc.",
            height=80
        )
    
    # Process button
    if st.button("🚀 Solve Problem", type="primary", use_container_width=True):
        if not problem_description:
            st.error("Please describe your data science problem.")
            return
        
        # Set LLM configuration in environment
        if 'selected_provider' in locals():
            os.environ["LLM_PROVIDER"] = selected_provider
        if 'selected_model' in locals() and selected_model:
            os.environ["LLM_MODEL"] = selected_model
        
        # Create streaming handler if enabled
        streaming_handler = None
        if 'show_streaming' in locals() and show_streaming:
            streaming_handler = MultiAgentAsyncHandler()
        
        # Results containers
        insights_container = st.container()
        code_container = st.container()
        results_container = st.container()
        
        # Create placeholders for progressive updates
        progress = st.progress(0)
        status = st.status("Initializing agent...", expanded=True)
        
        # Run the async process
        try:
            # Run in async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            with status:
                st.write("🔄 Processing your problem...")
                progress.progress(0.2)
                
                # Run the agent with selected provider and model
                selected_llm_provider = selected_provider if 'selected_provider' in locals() else None
                selected_llm_model = selected_model if 'selected_model' in locals() and selected_model else None
                
                result, handler = loop.run_until_complete(
                    process_problem(
                        problem_description, 
                        dataset_file, 
                        dataset_info, 
                        selected_llm_provider,
                        selected_llm_model,
                        streaming_handler
                    )
                )
                
                progress.progress(0.9)
                
                # Display results progressively
                if result.get('kaggle_insights'):
                    with insights_container:
                        display_kaggle_insights(result['kaggle_insights'])
                
                if result.get('generated_code') and result.get('code_explanation'):
                    with code_container:
                        # Parse code explanation if it's a JSON string
                        explanation = result['code_explanation']
                        if isinstance(explanation, str):
                            try:
                                explanation = json.loads(explanation)
                            except:
                                explanation = {"overview": explanation}
                        
                        display_code_with_explanation(
                            result['generated_code'],
                            explanation
                        )
                
                if result.get('generated_code'):
                    with results_container:
                        execution_results = result.get('execution_results', {})
                        display_execution_results(
                            execution_results.get('output', ''),
                            execution_results.get('error') if not execution_results.get('success') else None,
                            result.get('visualizations', [])
                        )
                
                progress.progress(1.0)
                status.update(label="✅ Analysis complete!", state="complete")
                
                # Display final summary if available
                if result.get('final_summary'):
                    st.divider()
                    st.markdown("### 📋 Summary")
                    st.info(result['final_summary'])
                
                # Display work directory information
                work_manager = get_work_dir_manager()
                if work_manager.work_dir:
                    st.divider()
                    st.markdown("### 📁 Work Directory")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Run ID:** `{work_manager.run_id}`")
                        st.markdown(f"**Location:** `{work_manager.work_dir}`")
                        
                    with col2:
                        st.markdown(f"**Results:** `{work_manager.results_dir}`")
                        if work_manager.log_file and work_manager.log_file.exists():
                            st.markdown(f"**Log:** `{work_manager.log_file}`")
                    
                    # Show files created
                    files_created = work_manager.list_files()
                    results_created = work_manager.list_files("results")
                    
                    if files_created or results_created:
                        st.markdown("**Files Created:**")
                        if files_created:
                            st.markdown("**Work directory:**")
                            for file in files_created:
                                st.markdown(f"• `{file}`")
                        
                        if results_created:
                            st.markdown("**Results directory:**")
                            for file in results_created:
                                st.markdown(f"• `{file}`")
                
                # Display agent activity if streaming was enabled
                if handler and hasattr(handler, 'format_messages_for_display'):
                    st.divider()
                    st.markdown("### 🔄 Agent Activity Log")
                    
                    messages = handler.format_messages_for_display()
                    if messages:
                        # Create a scrollable container for the activity log
                        activity_container = st.container()
                        with activity_container:
                            st.markdown("**Agent Activity Details:**")
                            # Display in a code block for better formatting
                            activity_text = "\n".join(messages)
                            st.text_area(
                                "Activity Log",
                                value=activity_text,
                                height=300,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                    else:
                        st.info("No agent activity captured during this run.")
                
        except Exception as e:
            status.update(label="❌ Error occurred", state="error")
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
        
        finally:
            loop.close()
    
    # Example problems
    with st.expander("💡 Example Problems"):
        st.markdown("""
        **Classification:**
        - Predict customer churn based on usage patterns
        - Classify emails as spam or not spam
        - Predict loan default risk
        
        **Regression:**
        - Predict house prices based on features
        - Forecast sales for next quarter
        - Estimate delivery time based on distance and traffic
        
        **Clustering:**
        - Segment customers based on purchasing behavior
        - Group similar products for recommendations
        
        **Time Series:**
        - Forecast stock prices
        - Predict energy consumption patterns
        """)


# Run the app
if __name__ == "__main__":
    main() 