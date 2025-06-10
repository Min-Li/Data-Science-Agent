"""
Streamlit Web Interface - Multi-Agent Data Science Application
=============================================================

This is the main user interface for the Data Science Agent system. It provides
a web-based interface for users to interact with the multi-agent workflow.

Core Features:
-------------
1. **Multi-Model Selection**: Choose different LLMs for each agent
2. **Dataset Upload**: Supports CSV, Excel, JSON files
3. **Real-time Progress**: Shows agent activities as they happen
4. **Interactive Approval**: Review and approve generated code
5. **Results Display**: Shows visualizations, metrics, and insights
6. **Work Directory**: All outputs saved in organized folders

UI Components:
-------------
- **Sidebar**: Model selection and configuration
- **Main Panel**: Problem input and dataset upload
- **Progress Display**: Real-time agent status updates
- **Results Section**: Summary, code, and visualizations
- **Download Options**: Save code, notebooks, and results

Agent Model Selection:
---------------------
Users can select different models for:
- Orchestrator: Planning and coordination (needs reasoning)
- Research: Kaggle search (needs speed and comprehension)
- Coding: Solution generation (needs code quality)

Work Directory Structure:
------------------------
Each run creates a timestamped directory:
```
agent_work_dir/multi-agent-{timestamp}/
├── inputs/          # Uploaded datasets
├── scripts/         # Generated Python scripts
├── results/         # Outputs, visualizations, models
├── execution.log    # Detailed execution log
└── metadata.json    # Run configuration
```

Key Functions:
-------------
- **create_sidebar_config()**: Model selection interface
- **run_agent()**: Main async workflow execution
- **create_initial_state()**: Prepares data for agents
- **display_final_results()**: Shows results with images

Safety & Transparency:
---------------------
- Shows exactly what code will run
- Requires user approval for local execution
- All agent decisions logged and visible
- Complete code visible before execution

Technical Details:
-----------------
- Uses Streamlit's async support for agent streaming
- Handles file uploads to work directory
- Manages PYTHONPATH for proper imports
- Serializes results for JSON storage
- Embeds images in markdown summaries

Common Issues:
-------------
- File upload size limits (Streamlit default: 200MB)
- Token limits for large datasets
- API rate limits with rapid iterations
- Memory usage with large vector databases

Interview Notes:
---------------
- This is the PRIMARY user interface - most users start here
- The multi-model selection is unique to this implementation
- Work directory organization ensures reproducibility
- The approval step has prevented many security issues
- Real-time streaming provides transparency into agent thinking
"""

import streamlit as st
import asyncio
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
from pathlib import Path

# Import components
from components.file_upload import create_upload_component

# Import agent system
import sys
sys.path.append('..')
from src.graph import create_data_science_graph
from src.state import DataScienceState
from src.utils import analyze_dataset, format_final_report
from src.llm_utils import (
    get_available_providers, 
    get_model_display_options, 
    get_recommended_model,
    AGENT_MODEL_RECOMMENDATIONS,
    MODEL_CONFIGS
)
from src.work_dir_manager import create_new_run, get_work_dir_manager, log_message

# Page configuration
st.set_page_config(
    page_title="Data Science Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

def make_serializable(obj: Any) -> Any:
    """Convert non-serializable objects to serializable format."""
    import json
    
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        # For objects with __dict__, try to extract relevant info
        try:
            # Special handling for common LangChain types
            if hasattr(obj, 'content'):
                return {"type": obj.__class__.__name__, "content": str(obj.content)}
            elif hasattr(obj, 'text'):
                return {"type": obj.__class__.__name__, "text": str(obj.text)}
            elif obj.__class__.__name__ in ['AIMessage', 'HumanMessage', 'SystemMessage']:
                # Handle LangChain message types
                content = getattr(obj, 'content', '')
                return {
                    "type": obj.__class__.__name__,
                    "content": str(content),
                    "additional_kwargs": getattr(obj, 'additional_kwargs', {})
                }
            else:
                return {"type": obj.__class__.__name__, "repr": str(obj)}
        except:
            return str(obj)
    else:
        # For all other types, try JSON serialization
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)

def main():
    """Main Streamlit application"""
    # Header
    st.title("🤖 Kaggle-Powered Data Science Agent")
    st.markdown(
        "Upload your dataset and describe your problem. "
        "The agent will search 647 Kaggle competitions and generate a solution!"
    )
    
    # Sidebar configuration
    with st.sidebar:
        create_sidebar_config()
    
    # Main interface - single column layout since results appear below
    # Dataset upload
    uploaded_file = create_upload_component()
    
    # Problem description
    problem = st.text_area(
        "Describe your problem:",
        placeholder="e.g., Predict customer churn based on usage patterns",
        height=100
    )
    
    # Run button
    if st.button("🚀 Generate Solution", type="primary", disabled=not uploaded_file):
        # Run the async function using asyncio
        try:
            with st.spinner("Running data science agent workflow..."):
                asyncio.run(run_agent(uploaded_file, problem))
        except Exception as e:
            st.error(f"Error running agent: {str(e)}")
            st.exception(e)

def create_sidebar_config():
    """Creates sidebar configuration options"""
    st.header("⚙️ Configuration")
    
    # Import model configurations
    sys.path.append('..')
    from src.llm_utils import (
        get_available_providers, 
        get_model_display_options, 
        get_recommended_model,
        AGENT_MODEL_RECOMMENDATIONS,
        MODEL_CONFIGS
    )
    
    # Model selection for each agent
    st.markdown("### 🤖 Agent Models")
    st.markdown("Choose different models for each agent type:")
    
    agent_models = {}
    
    # Orchestrator Model Selection
    st.markdown("#### 🧠 Orchestrator Agent")
    st.markdown("*Strategic planning and coordination*")
    
    orch_col1, orch_col2 = st.columns([1, 1])
    with orch_col1:
        orch_provider = st.selectbox(
            "Provider:",
            options=get_available_providers(),
            key="orch_provider",
            format_func=lambda x: MODEL_CONFIGS[x]["display_name"]
        )
    
    with orch_col2:
        orch_models = get_model_display_options(orch_provider)
        orch_recommended = get_recommended_model("orchestrator", orch_provider)
        orch_default_idx = list(orch_models.keys()).index(orch_recommended) if orch_recommended in orch_models else 0
        
        orch_model = st.selectbox(
            "Model:",
            options=list(orch_models.keys()),
            format_func=lambda x: orch_models[x],
            index=orch_default_idx,
            key="orch_model"
        )
    
    agent_models["orchestrator"] = {"provider": orch_provider, "model": orch_model}
    
    # Research Agent Model Selection
    st.markdown("#### 🔍 Research Agent") 
    st.markdown("*Information retrieval and search*")
    
    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        res_provider = st.selectbox(
            "Provider:",
            options=get_available_providers(),
            key="res_provider", 
            format_func=lambda x: MODEL_CONFIGS[x]["display_name"]
        )
    
    with res_col2:
        res_models = get_model_display_options(res_provider)
        res_recommended = get_recommended_model("research", res_provider)
        res_default_idx = list(res_models.keys()).index(res_recommended) if res_recommended in res_models else 0
        
        res_model = st.selectbox(
            "Model:",
            options=list(res_models.keys()),
            format_func=lambda x: res_models[x],
            index=res_default_idx,
            key="res_model"
        )
    
    agent_models["research"] = {"provider": res_provider, "model": res_model}
    
    # Coding Agent Model Selection
    st.markdown("#### 💻 Coding Agent")
    st.markdown("*Code generation and analysis*")
    
    cod_col1, cod_col2 = st.columns([1, 1])
    with cod_col1:
        cod_provider = st.selectbox(
            "Provider:",
            options=get_available_providers(),
            key="cod_provider",
            format_func=lambda x: MODEL_CONFIGS[x]["display_name"]
        )
    
    with cod_col2:
        cod_models = get_model_display_options(cod_provider)
        cod_recommended = get_recommended_model("coding", cod_provider)
        cod_default_idx = list(cod_models.keys()).index(cod_recommended) if cod_recommended in cod_models else 0
        
        cod_model = st.selectbox(
            "Model:",
            options=list(cod_models.keys()),
            format_func=lambda x: cod_models[x],
            index=cod_default_idx,
            key="cod_model"
        )
    
    agent_models["coding"] = {"provider": cod_provider, "model": cod_model}
    
    # Store model selection in session state for use in agent creation
    st.session_state["agent_models"] = agent_models
    
    # Display selected models summary
    with st.expander("📋 Selected Models Summary"):
        for agent_type, config in agent_models.items():
            provider_name = MODEL_CONFIGS[config["provider"]]["display_name"]
            model_desc = MODEL_CONFIGS[config["provider"]]["models"][config["model"]]["description"]
            st.markdown(f"**{agent_type.title()}:** {provider_name} - {model_desc}")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        max_iterations = st.slider("Max Iterations", 1, 30, 20, help="Number of iterations for the agent to refine the solution")
        temperature_override = st.slider("Temperature Override", 0.0, 1.0, 0.1, help="Override default model temperatures")
        enable_debug = st.checkbox("Enable Debug Mode", False)
        
        # Code execution settings
        st.markdown("**Code Execution:**")
        execution_mode = st.radio(
            "Mode:", 
            ["local", "remote"], 
            help="Local: runs on your machine with approval. Remote: uses Riza sandbox"
        )
        
        # Store advanced settings
        st.session_state["advanced_settings"] = {
            "max_iterations": max_iterations,
            "temperature_override": temperature_override if temperature_override != 0.1 else None,
            "debug_mode": enable_debug,
            "execution_mode": execution_mode
        }
    
    # Model capabilities info
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Provider Capabilities:**
        - **OpenAI**: Best overall performance, fast inference
        - **Anthropic**: Excellent reasoning, good code generation
        - **Google Gemini**: Strong multimodal, competitive performance  
        - **DeepSeek**: Cost-effective, good for reasoning tasks
        
        **Recommended Combinations:**
        - **Quality Focus**: GPT-4o + Claude Sonnet + GPT-4o
        - **Speed Focus**: GPT-4o-mini + Claude Haiku + Gemini Flash
        - **Cost-Effective**: DeepSeek across all agents
        """)
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This agent searches Kaggle competition solutions "
        "and generates custom code for your problem using "
        "your choice of AI models."
    )

@st.cache_data
def load_dataset(uploaded_file) -> pd.DataFrame:
    """Loads and caches uploaded dataset"""
    pass

async def run_agent(uploaded_file, problem: str):
    """Runs the data science agent workflow"""
    # Get model configurations from session state
    agent_models = st.session_state.get("agent_models", None)
    advanced_settings = st.session_state.get("advanced_settings", {})
    
    # Set debug mode if enabled
    if advanced_settings.get("debug_mode"):
        import os
        os.environ["DEBUG_MODE"] = "1"
    
    # Set execution mode
    if advanced_settings.get("execution_mode"):
        import os
        os.environ["CODE_EXECUTION_MODE"] = advanced_settings["execution_mode"]
    
    # Create work directory with multi-agent pattern
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = create_new_run(f"multi-agent-{timestamp}")
    work_manager = get_work_dir_manager()
    
    log_message(f"🚀 Starting multi-agent data science workflow")
    log_message(f"📝 Problem: {problem[:100]}...")
    if uploaded_file:
        log_message(f"📁 Dataset: {uploaded_file.name}")
    
    # Display work directory info in UI
    st.info(f"📁 Work directory created: `{work_dir}`")
    
    # Create initial state
    initial_state = create_initial_state(uploaded_file, problem, advanced_settings, work_manager)
    
    try:
        # Create agent graph with custom models
        st.info("🤖 Initializing multi-agent system...")
        graph = create_data_science_graph(agent_models=agent_models)
        
        # Show selected models
        if agent_models:
            model_info = []
            for agent_type, config in agent_models.items():
                provider_name = MODEL_CONFIGS[config["provider"]]["display_name"]
                model_desc = MODEL_CONFIGS[config["provider"]]["models"][config["model"]]["description"]
                model_info.append(f"**{agent_type.title()}:** {provider_name} - {config['model']}")
            st.info("🎯 Using custom models:\n" + "\n".join(model_info))
        
        st.info("🚀 Starting data science agent workflow...")
        
        # Configuration for the run
        config = {
            "configurable": {"thread_id": "streamlit_session"},
            "recursion_limit": 50
        }
        
        # Collect all results
        final_result = None
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        iteration_count = 0
        async for output in graph.astream(initial_state, config):
            iteration_count += 1
            
            # Update progress
            progress = min(0.9, iteration_count * 0.2)
            progress_bar.progress(progress)
            
            # Update status based on which node completed
            for node_name, node_state in output.items():
                if node_name == "orchestrator":
                    status_text.info("🧠 Orchestrator: Planning next steps...")
                elif node_name == "research":
                    status_text.info("🔍 Research: Searching Kaggle insights...")
                    if node_state.get("kaggle_insights"):
                        st.success(f"Found {len(node_state['kaggle_insights'])} Kaggle insights!")
                elif node_name == "coding":
                    status_text.info("💻 Coding: Generating and executing solution...")
                    if node_state.get("execution_results", {}).get("success"):
                        st.success("✅ Code executed successfully!")
                
                # Store the latest result
                final_result = node_state
        
        # Final progress
        progress_bar.progress(1.0)
        status_text.success("✅ Workflow completed!")
        
        # Display results
        if final_result:
            # DEBUG: Print what we have in final_result
            if advanced_settings.get("debug_mode"):
                import os
                os.environ["DEBUG_MODE"] = "1"
                print("🐛 [STREAMLIT] DEBUG: final_result keys:", list(final_result.keys()))
                print("🐛 [STREAMLIT] DEBUG: generated_code length:", len(final_result.get('generated_code', '')))
                print("🐛 [STREAMLIT] DEBUG: visualizations count:", len(final_result.get('visualizations', [])))
                print("🐛 [STREAMLIT] DEBUG: execution_results:", final_result.get('execution_results', {}))
            
            # Save results to work directory
            try:
                # Save Python script
                if final_result.get('generated_code'):
                    work_manager.save_result("final_solution.py", final_result['generated_code'])
                    log_message("💾 Saved final solution code")
                else:
                    log_message("⚠️ No generated code to save", "warning")
                    # DEBUG: print all keys to see what we have
                    log_message(f"Available keys in final_result: {list(final_result.keys())}", "info")
                
                # Save summary
                if final_result.get('final_summary'):
                    work_manager.save_result("summary.md", final_result['final_summary'])
                    log_message("💾 Saved final summary")
                
                # Save code explanation if available
                if final_result.get('code_explanation'):
                    work_manager.save_result("code_explanation.md", final_result['code_explanation'])
                    log_message("💾 Saved code explanation")
                
                # Save execution results
                if final_result.get('execution_results'):
                    exec_results = final_result['execution_results']
                    if exec_results.get('output'):
                        work_manager.save_result("execution_output.txt", exec_results['output'])
                        log_message("💾 Saved execution output")
                
                # Handle visualizations - copy from execution directory if they exist
                if work_manager.work_dir:
                    from pathlib import Path
                    import shutil
                    work_path = Path(work_manager.work_dir)
                    results_path = work_path / "results"
                    
                    # Look for visualization files in the work directory
                    viz_files_found = []
                    for viz_file in results_path.glob("visualization_*.png"):
                        if viz_file.exists():
                            viz_files_found.append(str(viz_file))
                    
                    if viz_files_found:
                        log_message(f"💾 Found {len(viz_files_found)} visualization files already saved")
                        for viz_file in viz_files_found:
                            log_message(f"📊 Visualization: {viz_file}")
                    else:
                        log_message("⚠️ No visualization files found in results directory", "warning")
                
                # Save complete results as JSON (with serialization fixes)
                import json
                from datetime import datetime
                
                # Create a serializable version of final_result
                serializable_result = make_serializable(final_result)
                
                results_json = {
                    "problem_description": problem,
                    "dataset_info": initial_state.get("dataset_info", {}),
                    "agent_models": agent_models,
                    "results": serializable_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                work_manager.save_result("complete_results.json", json.dumps(results_json, indent=2))
                log_message("💾 Saved complete results")
                
                work_manager.finalize_run("completed")
                
            except Exception as e:
                log_message(f"⚠️ Error saving results: {e}", "warning")
                import traceback
                log_message(f"Full error: {traceback.format_exc()}", "warning")
            
            display_final_results(final_result, work_manager)
        else:
            st.error("No results returned from agent workflow")
            work_manager.finalize_run("failed")
            
    except Exception as e:
        st.error(f"Error during workflow execution: {str(e)}")
        st.exception(e)
        work_manager.finalize_run("error")
    
    finally:
        # Display final work directory path
        if work_manager.work_dir:
            absolute_work_dir = str(Path(work_manager.work_dir).resolve())
            st.success(f"🎯 **Work directory**: `{absolute_work_dir}`")
            log_message(f"✅ Workflow completed. Work directory: {absolute_work_dir}")
        
        # Clean up temporary file if created (but not work directory files)
        temp_dataset_path = initial_state.get("dataset_path")
        if temp_dataset_path and not work_manager.work_dir:
            # Only clean up if we're not using work directory (fallback case)
            try:
                import os
                if Path(temp_dataset_path).exists():
                    os.unlink(temp_dataset_path)
            except:
                pass  # Ignore cleanup errors

def create_initial_state(uploaded_file, problem: str, advanced_settings: dict, work_manager) -> DataScienceState:
    """Creates initial state for agent workflow"""
    import tempfile
    import os
    
    # Handle uploaded file
    dataset_info = {"filename": uploaded_file.name if uploaded_file else "unknown"}
    dataset_path = None
    
    if uploaded_file:
        # Save uploaded file to work directory instead of temporary location
        try:
            import tempfile
            # First save to temp file, then copy to work directory
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_file_path = tmp.name
            
            # Copy to work directory inputs folder
            dataset_path = work_manager.copy_input_file(temp_file_path, uploaded_file.name)
            log_message(f"📁 Dataset saved to work directory: {uploaded_file.name}")
            
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
        except Exception as e:
            log_message(f"⚠️ Error saving to work directory, using temp file: {e}", "warning")
            # Fallback to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                dataset_path = tmp.name
        
        # Get relative path for execution (relative to work directory)
        work_dir_path = Path(work_manager.work_dir) if work_manager.work_dir else Path(".")
        try:
            dataset_relative_path = str(Path(dataset_path).relative_to(work_dir_path))
        except ValueError:
            # If not relative to work dir, use the absolute path
            dataset_relative_path = dataset_path
        
        dataset_info.update({
            "path": dataset_path,
            "relative_path": dataset_relative_path,
            "filename": uploaded_file.name
        })
        
        # Try to get basic dataset info
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                dataset_info.update({
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "preview": df.head(10).to_dict()  # Show top 10 rows to LLM
                })
                log_message(f"📊 Dataset analyzed: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.warning(f"Could not analyze dataset: {e}")
            log_message(f"⚠️ Could not analyze dataset: {e}", "warning")
    
    return {
        "problem_description": problem,
        "dataset_info": dataset_info,
        "dataset_path": dataset_path,
        "max_iterations": advanced_settings.get("max_iterations", 3),
        "current_iteration": 0,
        "search_queries": [],
        "kaggle_insights": [],
        "generated_code": "",
        "code_explanation": "",
        "execution_results": {},
        "error_messages": [],
        "visualizations": [],
        "final_summary": "",
        "messages": [],
        "work_dir": str(work_manager.work_dir) if work_manager.work_dir else None
    }

def update_ui_with_chunk(chunk: Dict, progress, status):
    """Updates UI with agent progress"""
    for node_name, node_state in chunk.items():
        if node_name == "orchestrator":
            progress.progress(0.2)
            status.update(label="🧠 Orchestrator planning next steps...", state="running")
        elif node_name == "research":
            progress.progress(0.5)
            status.update(label="🔍 Research agent searching Kaggle insights...", state="running")
        elif node_name == "coding":
            progress.progress(0.8)
            status.update(label="💻 Coding agent generating solution...", state="running")

def display_final_results(final_state: DataScienceState, work_manager=None):
    """Displays final solution and results"""
    st.success("✅ Agent workflow completed!")
    
    # Display work directory information first
    if work_manager and work_manager.work_dir:
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
    
    # Display final summary
    if final_state.get("final_summary"):
        st.markdown("### 📋 Solution Summary")
        
        # Check if summary contains image references and we have a work directory
        if work_manager and work_manager.work_dir and "![" in final_state["final_summary"]:
            # The summary contains image references - need to render them properly
            from pathlib import Path
            work_path = Path(work_manager.work_dir)
            results_path = work_path / "results"
            
            # Find all image references in markdown
            import re
            img_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
            
            # Extract image references for later
            image_refs = []
            for match in re.finditer(img_pattern, final_state["final_summary"]):
                alt_text = match.group(1)
                img_path = match.group(2)
                image_refs.append((alt_text, img_path))
            
            # Split summary into sections
            summary_parts = final_state["final_summary"].split("## 📊 Visualizations")
            
            if len(summary_parts) > 1:
                # Display everything before visualizations section
                text_before_viz = summary_parts[0]
                # Remove any image markdown from this part
                text_before_viz = re.sub(img_pattern, '', text_before_viz)
                st.markdown(text_before_viz)
                
                # Now display visualizations section with actual images
                st.markdown("## 📊 Visualizations")
                
                # Get the visualization section content
                viz_section = summary_parts[1].split("## 🔑 Key Findings")[0] if "## 🔑 Key Findings" in summary_parts[1] else summary_parts[1].split("## 🎯 Summary")[0]
                
                # Extract visualization descriptions
                viz_lines = viz_section.strip().split('\n')
                current_viz = None
                
                for line in viz_lines:
                    if line.startswith("### "):
                        current_viz = line
                        st.markdown(line)
                    elif line.startswith("*") and line.endswith("*"):
                        st.markdown(line)
                    elif "![" in line:
                        # This is an image reference - display the actual image
                        match = re.search(img_pattern, line)
                        if match:
                            alt_text = match.group(1)
                            img_path = match.group(2)
                            
                            # Handle relative paths
                            if img_path.startswith("../"):
                                full_path = work_path / img_path[3:]
                            else:
                                full_path = results_path / img_path
                            
                            # Display image if it exists
                            if full_path.exists():
                                try:
                                    from PIL import Image
                                    img = Image.open(full_path)
                                    st.image(img, caption=alt_text, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not display image {img_path}: {e}")
                            else:
                                st.warning(f"Image not found: {img_path}")
                
                # Display remaining sections (Key Findings, Summary)
                if "## 🔑 Key Findings" in summary_parts[1]:
                    remaining = "## 🔑 Key Findings" + summary_parts[1].split("## 🔑 Key Findings")[1]
                    st.markdown(remaining)
                elif "## 🎯 Summary" in summary_parts[1]:
                    remaining = "## 🎯 Summary" + summary_parts[1].split("## 🎯 Summary")[1]
                    st.markdown(remaining)
            else:
                # No visualizations section - display as is without image references
                summary_text = re.sub(img_pattern, '', final_state["final_summary"])
                st.markdown(summary_text)
        else:
            # No images or no work directory - just display as markdown
            st.markdown(final_state["final_summary"])
    
    # Display generated code
    if final_state.get("generated_code"):
        st.markdown("### 💻 Generated Code")
        st.code(final_state["generated_code"], language="python")
        
        # Download button for code
        st.download_button(
            label="📥 Download Code",
            data=final_state["generated_code"],
            file_name="data_science_solution.py",
            mime="text/x-python"
        )
    
    # Display execution results
    if final_state.get("execution_results"):
        exec_results = final_state["execution_results"]
        st.markdown("### 🚀 Execution Results")
        
        if exec_results.get("success"):
            st.success("Code executed successfully!")
            if exec_results.get("output"):
                st.markdown("**Output:**")
                st.text(exec_results["output"])
        else:
            st.error("Code execution failed!")
            if exec_results.get("error"):
                st.error(exec_results["error"])
    
    # Display visualizations (now handled in summary section with actual files)
    # The visualizations are displayed as part of the summary above
    
    # Display research insights if any
    if final_state.get("kaggle_insights"):
        st.markdown("### 🔍 Kaggle Insights")
        insights = final_state["kaggle_insights"]
        for i, insight in enumerate(insights[:3]):  # Show top 3
            with st.expander(f"Insight {i+1}: {insight.get('title', 'Unknown')}"):
                st.write(f"**Score:** {insight.get('score', 0):.3f}")
                # Only show summary if it exists and is not 'N/A' or 'Unknown'
                if insight.get('summary') and insight['summary'] not in ['N/A', 'Unknown']:
                    st.write(f"**Summary:** {insight['summary']}")
                # Show ML techniques if available
                if insight.get('ml_techniques'):
                    st.write(f"**Techniques:** {', '.join(insight['ml_techniques'][:5])}")
                elif insight.get('techniques'):
                    st.write(f"**Techniques:** {', '.join(insight['techniques'])}")
    
    # Progress completed
    st.balloons()

if __name__ == "__main__":
    main() 