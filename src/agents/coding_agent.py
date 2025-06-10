"""
Coding Agent - Python Code Generation and Execution
==================================================

This module implements the coding agent that generates complete, executable Python
solutions for data science problems. It takes insights from the research agent and
transforms them into working code with visualizations and analysis.

Core Responsibilities:
---------------------
1. **Code Generation**: Creates complete Python scripts based on problem + insights
2. **Auto-fixing**: Attempts to fix errors automatically (up to 3 tries)
3. **Execution**: Runs code locally (with approval) or remotely (Riza sandbox)
4. **Result Capture**: Collects outputs, metrics, and visualizations
5. **Iterative Improvement**: Learns from errors to generate better code

Code Generation Strategy:
------------------------
- Generates COMPLETE, self-contained scripts (no hidden code injection)
- Includes all imports, data loading, analysis, and visualization
- Saves outputs with descriptive filenames (not generic "figure1.png")
- Prints clear summaries of what was done and results achieved
- Uses only standard libraries (pandas, numpy, sklearn, matplotlib, seaborn)

Key Features:
------------
- **Transparent Architecture**: Agent sees and controls ALL executed code
- **Smart Naming**: Files named based on content (e.g., "correlation_heatmap.png")
- **Error Context**: Previous failures passed to improve next attempt
- **Multiple Scripts**: Can generate exploration, training, evaluation scripts
- **Work Directory Aware**: Saves all outputs to organized directories

Technical Components:
--------------------
- **CodeSolution**: Pydantic model for structured code output
- **generate_code_solution()**: Main code generation with LLM
- **execute_and_iterate()**: Execution loop with auto-retry
- **fix_code_error()**: Targeted fixes for common issues
- **extract_code_from_response()**: Robust code extraction from LLM output

Common Patterns Generated:
-------------------------
1. Data exploration with statistics and visualizations
2. Feature engineering and preprocessing
3. Model training with cross-validation
4. Hyperparameter tuning with grid search
5. Result evaluation with metrics and plots

Interview Notes:
---------------
- This is where the "magic" happens - converts ideas to working code
- The transparent architecture means NO hidden code manipulation
- Quality depends heavily on the LLM model used (GPT-4 recommended)
- The agent is responsible for EVERYTHING - imports, execution, saving
- Debug mode shows the complete generated code before execution
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import os
import logging

from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from state import DataScienceState
from prompts import CODING_AGENT_PROMPT, CODE_GENERATION_TEMPLATE
from tools.code_executor import create_code_executor, BaseCodeExecutor
from llm_utils import get_default_llm

# Set up logger
logger = logging.getLogger(__name__)

def debug_print(message: str):
    """Debug print that only shows in debug mode."""
    if os.getenv("DEBUG_MODE"):
        print(f"🐛 [CODING] {message}")
    logger.debug(f"[CODING] {message}")


class CodeSolution(BaseModel):
    """Structured output for generated code solution."""
    script_name: str = Field(description="Descriptive filename for the script (e.g., 'data_exploration.py')")
    approach: str = Field(description="Brief description of what this script does")
    code: str = Field(description="Complete Python code for this script")
    expected_output: str = Field(description="What this script is expected to produce")
    libraries_used: List[str] = Field(description="List of libraries used in this script")


class CodeExplanation(BaseModel):
    """Structured explanation of the code solution."""
    overview: str = Field(description="High-level overview of the solution")
    key_steps: List[str] = Field(description="Key steps in the solution")
    techniques: List[str] = Field(description="ML/DS techniques used")
    assumptions: List[str] = Field(description="Assumptions made in the solution")


async def generate_code_solution(
    problem_description: str,
    dataset_info: str,
    kaggle_insights: List[Dict[str, Any]],
    llm,
    work_dir: str = None,
    previous_execution_results: Dict[str, Any] = None,
    previous_errors: List[str] = None,
    iteration: int = 1,
    previous_scripts: List[str] = None,
    previous_outputs: str = ""
) -> Tuple[str, str, CodeExplanation]:
    """
    Generate Python code to solve the data science problem.
    
    Args:
        problem_description: Description of the problem
        dataset_info: Information about the dataset
        kaggle_insights: Insights from Kaggle research
        llm: Language model for code generation
    
    Returns:
        Tuple of (generated_code, explanation)
    """
    # Format insights for the prompt
    insights_text = ""
    if kaggle_insights:
        for insight in kaggle_insights[:5]:  # Use top 5 insights
            insights_text += f"\n\n**{insight.get('competition_name', 'Competition')}**\n"
            insights_text += f"- Problem Type: {insight.get('problem_type', 'Unknown')}\n"
            insights_text += f"- Key Techniques: {', '.join(insight.get('ml_techniques', []))}\n"
            insights_text += f"- Insights: {insight.get('key_insights', 'N/A')}\n"
    else:
        insights_text = "No specific Kaggle insights available. Using general best practices."
    
    # Create the code generation prompt
    # Extract dataset path from dataset_info
    dataset_path = ""
    if isinstance(dataset_info, dict):
        dataset_path = dataset_info.get('path', 'dataset.csv')
    else:
        dataset_path = str(dataset_info)
    
    # Build error context from previous execution failures
    error_context = ""
    if previous_execution_results and not previous_execution_results.get("success", True):
        error_context += f"\n\n⚠️ **PREVIOUS EXECUTION FAILED**:\n"
        error_context += f"Error: {previous_execution_results.get('error', 'Unknown error')}\n"
        # Include the actual output for debugging
        if previous_execution_results.get('output'):
            error_context += f"\nFull output:\n{previous_execution_results.get('output')}\n"
        
    if previous_errors:
        if not error_context:
            error_context += "\n\n⚠️ **PREVIOUS ERRORS**:\n"
        for i, error in enumerate(previous_errors[-2:], 1):  # Show last 2 errors
            error_context += f"Error {i}: {error}\n"
            
    if error_context:
        error_context += "\n🔧 **IMPORTANT**: Please fix these issues in your new code solution!\n"
        error_context += "- Use only standard libraries (pandas, numpy, sklearn, matplotlib, seaborn)\n"
        error_context += "- Avoid libraries like yellowbrick, plotly, or other specialized visualization libraries\n"
        error_context += "- Handle missing dependencies gracefully\n"
    
    # Format previous scripts and outputs
    previous_scripts_text = ", ".join(previous_scripts) if previous_scripts else "None"
    
    prompt = CODE_GENERATION_TEMPLATE.format(
        problem=problem_description,
        dataset_path=dataset_path,
        work_dir=work_dir or ".",
        insights=insights_text,
        error_context=error_context,
        iteration=iteration,
        previous_scripts=previous_scripts_text,
        previous_outputs=previous_outputs or "No previous outputs yet"
    )
    
    # Create parser for structured output
    parser = PydanticOutputParser(pydantic_object=CodeSolution)
    
    # Generate code with structured output
    messages = [
        SystemMessage(content=CODING_AGENT_PROMPT),
        HumanMessage(content=f"{prompt}\n\n{parser.get_format_instructions()}")
    ]
    
    response = await llm.ainvoke(messages)
    
    # Parse the response
    try:
        solution = parser.parse(response.content)
        debug_print(f"Successfully parsed structured output")
    except Exception as e:
        debug_print(f"Failed to parse structured output: {e}")
        
        # DEBUG: Print full response in debug mode
        if os.getenv("DEBUG_MODE"):
            debug_print("=== FULL CODING LLM RESPONSE FOR DEBUGGING ===")
            debug_print(response.content)
            debug_print("=== END FULL CODING RESPONSE ===")
        
        # Try to extract JSON manually
        try:
            import json
            import re
            # Look for JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                debug_print(f"Found JSON match: {json_match.group(0)[:200]}...")
                json_data = json.loads(json_match.group(0))
                # Extract from nested structure if needed
                if 'properties' in json_data and 'code' in json_data['properties']:
                    code = json_data['properties']['code']
                elif 'code' in json_data:
                    code = json_data['code']
                else:
                    raise ValueError("No code field found in JSON")
                
                solution = CodeSolution(
                    script_name=json_data.get('properties', {}).get('script_name', json_data.get('script_name', f'script_{iteration}.py')),
                    approach=json_data.get('properties', {}).get('approach', json_data.get('approach', 'Manual extraction')),
                    code=code,
                    expected_output=json_data.get('properties', {}).get('expected_output', json_data.get('expected_output', 'See code output')),
                    libraries_used=json_data.get('properties', {}).get('libraries_used', json_data.get('libraries_used', extract_libraries(code)))
                )
                debug_print(f"Successfully extracted code from JSON manually")
            else:
                raise ValueError("No JSON found in response")
        except Exception as e2:
            debug_print(f"Manual JSON extraction also failed: {e2}")
            # Final fallback to extracting code from response
            code = extract_code_from_response(response.content)
            solution = CodeSolution(
                script_name=f"script_{iteration}.py",
                approach="Extracted from response",
                code=code,
                expected_output="See code output",
                libraries_used=extract_libraries(code)
            )
    
    # Create explanation
    explanation = CodeExplanation(
        overview=solution.approach,
        key_steps=extract_key_steps(solution.code),
        techniques=extract_techniques(solution.code, kaggle_insights),
        assumptions=["Dataset is properly formatted", "Target variable is identified"]
    )
    
    return solution.script_name, solution.code, explanation


async def execute_and_iterate(
    code: str,
    code_executor: BaseCodeExecutor,
    max_attempts: int = 3,
    llm = None
) -> Dict[str, Any]:
    """
    Execute code and iterate on errors if needed.
    
    Args:
        code: Python code to execute
        code_executor: Code execution tool
        max_attempts: Maximum number of attempts to fix errors
        llm: Language model for fixing errors (if provided)
    
    Returns:
        Execution results
    """
    attempt = 0
    current_code = code
    
    while attempt < max_attempts:
        result = code_executor._run(current_code)
        
        if result["success"]:
            return result
        
        # If we have an LLM and there's an error, try to fix it
        if llm and result.get("error") and attempt < max_attempts - 1:
            print(f"[Coding Agent] Execution failed, attempting to fix error (attempt {attempt + 2}/{max_attempts})...")
            # Pass both error and output for better context
            error_with_output = result["error"]
            if result.get("output"):
                error_with_output += f"\n\nFull output:\n{result['output']}"
            current_code = await fix_code_error(
                current_code, 
                error_with_output, 
                llm
            )
            attempt += 1
        else:
            return result
    
    return result


async def fix_code_error(code: str, error: str, llm) -> str:
    """
    Attempt to fix code based on error message.
    
    Args:
        code: Original code
        error: Error message
        llm: Language model
    
    Returns:
        Fixed code
    """
    # Add specific guidance for common dependency errors
    fix_guidance = ""
    if "ModuleNotFoundError" in error or "No module named" in error:
        if "yellowbrick" in error:
            fix_guidance = """
🔧 SPECIFIC FIX NEEDED: The error shows yellowbrick is not available.
- Replace yellowbrick.cluster.KElbowVisualizer with manual elbow method using matplotlib
- Use manual loop to test different K values and plot results
- Use sklearn.metrics.silhouette_score for evaluation instead of yellowbrick tools
"""
        elif "plotly" in error:
            fix_guidance = """
🔧 SPECIFIC FIX NEEDED: Replace plotly with matplotlib for visualization.
"""
        else:
            fix_guidance = """
🔧 SPECIFIC FIX NEEDED: Use only standard libraries:
- pandas, numpy, sklearn, matplotlib, seaborn
- Avoid specialized libraries that may not be installed
"""
    
    fix_prompt = f"""
Fix the following Python code error:

Error:
{error}

{fix_guidance}

Code:
```python
{code}
```

IMPORTANT: Only use standard libraries (pandas, numpy, sklearn, matplotlib, seaborn).
Do not use yellowbrick, plotly, or other specialized libraries.

Provide the corrected code only, without explanations.
"""
    
    messages = [
        SystemMessage(content="You are a Python expert. Fix the code to resolve the error. Use only standard libraries."),
        HumanMessage(content=fix_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    
    # Extract code from response
    fixed_code = extract_code_from_response(response.content)
    return fixed_code if fixed_code else code


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response."""
    import re
    
    # First, try to find code in JSON structure
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            import json
            json_str = json_match.group(0)
            if os.getenv("DEBUG_MODE"):
                debug_print(f"extract_code_from_response: Found JSON: {json_str[:200]}...")
            json_data = json.loads(json_str)
            if 'properties' in json_data and 'code' in json_data['properties']:
                return json_data['properties']['code'].replace('\\n', '\n')
            elif 'code' in json_data:
                return json_data['code'].replace('\\n', '\n')
    except Exception as e:
        if os.getenv("DEBUG_MODE"):
            debug_print(f"extract_code_from_response: JSON parsing failed: {e}")
        pass
    
    # Look for code blocks with ```python
    code_pattern = r'```python\n(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Look for code blocks with just ```
    code_pattern = r'```\n(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If no code blocks, check if the response looks like code
    lines = response.strip().split('\n')
    code_indicators = ['import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'print(', 'plt.']
    
    # If the response has code-like patterns, return it
    if any(any(line.strip().startswith(indicator) for indicator in code_indicators) for line in lines):
        return response.strip()
    
    # Final fallback - return the response as-is
    return response.strip()


def extract_libraries(code: str) -> List[str]:
    """Extract imported libraries from code."""
    import re
    libraries = set()
    
    # Match import statements
    import_pattern = r'^\s*(?:from\s+(\S+)|import\s+(\S+))'
    
    for line in code.split('\n'):
        match = re.match(import_pattern, line)
        if match:
            lib = match.group(1) or match.group(2)
            if lib:
                # Get base library name
                base_lib = lib.split('.')[0]
                libraries.add(base_lib)
    
    return list(libraries)


def extract_key_steps(code: str) -> List[str]:
    """Extract key steps from code based on comments and structure."""
    steps = []
    
    # Look for comments that describe steps
    lines = code.split('\n')
    for line in lines:
        if line.strip().startswith('#') and len(line.strip()) > 10:
            comment = line.strip()[1:].strip()
            if comment and not comment.startswith('-'):
                steps.append(comment)
    
    # If no comments, generate basic steps
    if not steps:
        if 'train_test_split' in code:
            steps.append("Split data into training and testing sets")
        if 'fit(' in code:
            steps.append("Train the model")
        if 'predict(' in code:
            steps.append("Make predictions")
        if 'score(' in code or 'accuracy_score' in code:
            steps.append("Evaluate model performance")
    
    return steps[:5]  # Return top 5 steps


def extract_techniques(code: str, insights: List[Dict[str, Any]]) -> List[str]:
    """Extract ML/DS techniques used in the code."""
    techniques = set()
    
    # Common technique patterns
    technique_patterns = {
        'RandomForest': 'Random Forest',
        'XGBoost': 'XGBoost',
        'GradientBoosting': 'Gradient Boosting',
        'LogisticRegression': 'Logistic Regression',
        'LinearRegression': 'Linear Regression',
        'SVM': 'Support Vector Machine',
        'KMeans': 'K-Means Clustering',
        'StandardScaler': 'Feature Scaling',
        'PCA': 'Principal Component Analysis',
        'cross_val': 'Cross Validation',
        'GridSearchCV': 'Hyperparameter Tuning',
        'train_test_split': 'Train-Test Split'
    }
    
    for pattern, technique in technique_patterns.items():
        if pattern in code:
            techniques.add(technique)
    
    # Add techniques from insights if they appear relevant
    for insight in insights[:3]:
        for tech in insight.get('ml_techniques', []):
            if any(keyword in code.lower() for keyword in tech.lower().split()):
                techniques.add(tech)
    
    return list(techniques)


def create_coding_node(callbacks=None, custom_model=None):
    """
    Create the coding agent node for the graph.
    
    This node:
    1. Generates Python code based on research insights
    2. Executes the code in a sandboxed environment
    3. Captures results and visualizations
    4. Handles errors and iterations
    
    Args:
        callbacks: Optional callbacks for streaming
        custom_model: Dict with custom model config:
                     {"provider": "openai", "model": "gpt-4o"}
    """
    async def coding_agent(state: DataScienceState) -> Dict[str, Any]:
        print("[Coding Agent] Generating Python solution based on insights...")
        debug_print("Coding agent started")
        debug_print(f"Input state keys: {list(state.keys())}")
        debug_print(f"State type: {type(state)}")
        
        # Initialize LLM - use custom model if provided
        debug_print("Initializing LLM for coding agent...")
        if custom_model:
            debug_print(f"Using custom model from parameters: {custom_model}")
            from llm_utils import create_custom_llm
            llm = create_custom_llm(
                provider=custom_model["provider"],
                model_id=custom_model["model"],
                temperature=0.2,
                streaming=True,
                callbacks=callbacks,
                agent_name="💻 Coding",
                max_tokens=16000  # Increase from default 1024
            )
        elif os.getenv("CODING_PROVIDER") and os.getenv("CODING_MODEL"):
            # Check for command-line environment variables
            provider = os.getenv("CODING_PROVIDER")
            model = os.getenv("CODING_MODEL")
            debug_print(f"Using custom model from environment: {provider}/{model}")
            from llm_utils import create_custom_llm
            llm = create_custom_llm(
                provider=provider,
                model_id=model,
                temperature=0.2,
                streaming=True,
                callbacks=callbacks,
                agent_name="💻 Coding",
                max_tokens=16000  # Increase from default 1024
            )
        else:
            debug_print("Using default LLM")
            llm = get_default_llm(temperature=0.2, streaming=True, callbacks=callbacks)
        debug_print("LLM initialized successfully")
        
        # Generate code solution
        problem_desc = state.get("problem_description", "")
        dataset_info = state.get("dataset_info", {})
        insights = state.get("kaggle_insights", [])
        
        # Get previous execution results and errors for context
        previous_execution_results = state.get("execution_results", {})
        previous_errors = state.get("error_messages", [])
        
        debug_print(f"Problem description: {problem_desc[:100]}...")
        debug_print(f"Dataset info: {dataset_info}")
        debug_print(f"Kaggle insights: {len(insights)} insights available")
        debug_print(f"Previous execution results: {previous_execution_results}")
        debug_print(f"Previous errors: {previous_errors}")
        
        # Check if this is a retry after an error
        is_retry = bool(previous_execution_results and not previous_execution_results.get("success", True))
        if is_retry:
            print(f"[Coding Agent] 🔄 Retrying after previous error: {previous_execution_results.get('error', 'Unknown')[:100]}...")
        
        # Get info about previous scripts for context
        current_iteration = state.get("current_iteration", 0) + 1
        created_scripts = state.get("created_scripts", [])
        script_outputs = state.get("script_outputs", {})
        
        # Format previous outputs summary
        previous_scripts = [s["name"] for s in created_scripts]
        previous_outputs_summary = ""
        for script in created_scripts[-3:]:  # Show last 3 scripts
            script_name = script["name"]
            if script_name in script_outputs:
                output = script_outputs[script_name][:500]  # First 500 chars
                previous_outputs_summary += f"\n\n**{script_name} output:**\n{output}..."
        
        # Get work directory from state
        work_dir = state.get("work_dir", None)
        debug_print(f"Work directory from state: {work_dir}")
        
        debug_print("Generating code solution...")
        script_name, code, explanation = await generate_code_solution(
            problem_description=problem_desc,
            dataset_info=dataset_info,
            kaggle_insights=insights,
            llm=llm,
            work_dir=work_dir,
            previous_execution_results=previous_execution_results,
            previous_errors=previous_errors,
            iteration=current_iteration,
            previous_scripts=previous_scripts,
            previous_outputs=previous_outputs_summary
        )
        debug_print(f"Script name: {script_name}")
        debug_print(f"Code generated, length: {len(code)} characters")
        debug_print(f"Code solution created successfully")
        
        print(f"[Coding Agent] Generated script '{script_name}' using: {', '.join(explanation.techniques[:3])}")
        
        # Create code executor with dataset
        dataset_path = dataset_info.get('path') if isinstance(dataset_info, dict) else None
        debug_print(f"Dataset path: {dataset_path}")
        
        debug_print("Creating code executor...")
        # Use local execution by default for convenience and speed
        # Set to "remote" if you prefer Riza's isolated environment
        execution_mode = os.getenv("CODE_EXECUTION_MODE", "local")
        debug_print(f"Code execution mode: {execution_mode}")
        
        # Pass work_dir to executor if available
        from tools.code_executor import LocalCodeExecutor, RizaCodeExecutor
        if execution_mode == "local":
            code_executor = LocalCodeExecutor(dataset_path=dataset_path)
            if work_dir:
                code_executor.work_dir = work_dir
                debug_print(f"Set executor work_dir to: {work_dir}")
        else:
            code_executor = create_code_executor(dataset_path=dataset_path, execution_mode=execution_mode)
        
        debug_print("Code executor created")
        
        # Save the script to work directory
        script_path = None
        if work_dir:
            from pathlib import Path
            work_path = Path(work_dir)
            scripts_dir = work_path / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            
            script_path = scripts_dir / script_name
            with open(script_path, 'w') as f:
                f.write(code)
            
            debug_print(f"Saved script to: {script_path}")
            print(f"[Coding Agent] 💾 Saved script: {script_name}")
        else:
            debug_print("No work directory available for saving script")
        
        # Execute code with retry logic
        print(f"[Coding Agent] Executing '{script_name}' in sandbox environment...")
        debug_print("Starting code execution with retry logic...")
        result = await execute_and_iterate(
            code=code,
            code_executor=code_executor,
            max_attempts=3,
            llm=llm
        )
        debug_print(f"Code execution completed: success={result.get('success', False)}")
        
        # Update state with results
        update_dict = {
            "generated_code": code,
            "code_explanation": json.dumps({
                "script_name": script_name,
                "overview": explanation.overview,
                "key_steps": explanation.key_steps,
                "techniques": explanation.techniques,
                "assumptions": explanation.assumptions
            })
        }
        
        # Track created scripts
        new_scripts = list(created_scripts)
        if script_path:
            new_scripts.append({
                "name": script_name,
                "path": str(script_path)
            })
        update_dict["created_scripts"] = new_scripts
        
        if result["success"]:
            print("[Coding Agent] ✅ Code executed successfully!")
            update_dict.update({
                "execution_results": {
                    "output": result["output"],
                    "data": result.get("data", {}),
                    "success": True
                },
                "visualizations": result.get("visualizations", [])
            })
            
            # Add performance metrics if found
            if result.get("data"):
                update_dict["execution_results"]["metrics"] = result["data"]
            
            # Save script output
            new_script_outputs = dict(script_outputs)
            if result.get("output"):
                new_script_outputs[script_name] = result["output"]
            update_dict["script_outputs"] = new_script_outputs
        else:
            print(f"[Coding Agent] ❌ Code execution failed: {result.get('error', 'Unknown error')[:100]}...")
            
            # Accumulate error messages instead of overwriting
            existing_errors = state.get("error_messages", [])
            new_error = result.get("error", "Unknown error")
            all_errors = existing_errors + [new_error] if new_error not in existing_errors else existing_errors
            
            update_dict.update({
                "execution_results": {
                    "success": False, 
                    "error": new_error,
                    "output": result.get("output", "")  # Include the output for debugging
                },
                "error_messages": all_errors,
                "visualizations": []
            })
        
        # Increment iteration counter
        update_dict["current_iteration"] = state.get("current_iteration", 0) + 1
        
        return update_dict
    
    return coding_agent


# Convenience function for standalone use
async def run_coding_agent(
    problem_description: str,
    dataset_info: str,
    kaggle_insights: List[Dict[str, Any]],
    dataset_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the coding agent standalone.
    
    Args:
        problem_description: Problem to solve
        dataset_info: Dataset information
        kaggle_insights: Insights from research
        dataset_path: Optional path to dataset file
    
    Returns:
        Dictionary with code, explanation, and execution results
    """
    # Create a minimal state
    state = DataScienceState(
        problem_description=problem_description,
        dataset_info=dataset_info if isinstance(dataset_info, dict) else {"description": dataset_info},
        kaggle_insights=kaggle_insights,
        current_iteration=0
    )
    
    # If dataset_path is provided, add it to dataset_info
    if dataset_path and isinstance(state.dataset_info, dict):
        state.dataset_info['path'] = dataset_path
    
    # Get the node function and run it (will use local execution by default)
    coding_node = create_coding_node()
    result = await coding_node(state)
    
    return {
        "code": result.get("generated_code"),
        "explanation": result.get("code_explanation"),
        "execution_result": result.get("execution_results"),
        "error": result.get("error_messages"),
        "visualizations": result.get("visualizations", [])
    } 