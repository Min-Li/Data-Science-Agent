"""
Orchestrator Agent - The Brain of the Multi-Agent System
=======================================================

This module implements the orchestrator agent that acts as the central coordinator
for the entire data science workflow. Think of it as a project manager that decides
what needs to be done next and delegates work to specialized agents.

Key Responsibilities:
--------------------
1. **Problem Analysis**: Understands the user's data science problem
2. **Strategic Planning**: Decides whether to research, code, or conclude
3. **Task Delegation**: Routes work to Research or Coding agents
4. **Progress Monitoring**: Tracks iterations and prevents infinite loops
5. **Result Evaluation**: Determines if the solution is satisfactory
6. **Summary Generation**: Creates comprehensive final reports with metrics

Decision Logic:
--------------
- No Kaggle insights yet? → Send to Research Agent
- Have insights but no code? → Send to Coding Agent
- Code executed with errors? → Analyze and decide (retry or different approach)
- Good results or max iterations? → End workflow with summary

Key Components:
--------------
- **OrchestratorPlan**: Structured output with next_action, reasoning, instructions
- **get_orchestrator_plan()**: LLM interaction with JSON/text fallback
- **create_summary()**: Generates rich markdown summaries with visualizations
- **Iteration Control**: Max 20 iterations by default to prevent runaway

Technical Features:
------------------
- Supports custom LLM models per agent (OpenAI, Anthropic, etc.)
- Graceful fallback when structured output isn't supported
- Debug mode for detailed execution tracing
- State preservation for all agent outputs

Interview Notes:
---------------
- Uses LangGraph's Command pattern for routing (goto parameter)
- Implements supervisor pattern from open_deep_research
- The orchestrator makes ALL routing decisions - other agents just execute
- Critical for preventing infinite loops with iteration counting
- The summary generation extracts metrics from execution output using regex
"""

from typing import Dict, Any, List, Literal
import os
import logging
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
from langgraph.graph import END
from pydantic import BaseModel, Field
import pandas as pd

from state import DataScienceState
from prompts import ORCHESTRATOR_SYSTEM_PROMPT
from llm_utils import get_default_llm

# Set up logger
logger = logging.getLogger(__name__)

def debug_print(message: str):
    """Debug print that only shows in debug mode."""
    if os.getenv("DEBUG_MODE"):
        print(f"🐛 [ORCHESTRATOR] {message}")
    logger.debug(f"[ORCHESTRATOR] {message}")

class OrchestratorPlan(BaseModel):
    """Structured output for orchestrator planning"""
    next_action: Literal["research", "coding", "end"] = Field(
        description="Next action to take in the workflow"
    )
    reasoning: str = Field(
        description="Reasoning for the decision"
    )
    instructions: str = Field(
        description="Specific instructions for the next agent"
    )

class ProblemAnalysis(BaseModel):
    """Structured analysis of the data science problem"""
    problem_type: Literal["classification", "regression", "clustering", "time_series", "other"] = Field(
        description="Type of data science problem"
    )
    data_requirements: List[str] = Field(
        description="Key data requirements extracted from the problem"
    )
    suggested_approaches: List[str] = Field(
        description="Initial thoughts on approaches to try"
    )


async def get_orchestrator_plan(llm, context: str) -> OrchestratorPlan:
    """
    Get orchestrator plan with fallback for models without structured output support.
    
    Args:
        llm: Language model instance
        context: Current workflow context
        
    Returns:
        OrchestratorPlan: The planning decision
    """
    debug_print("Building messages for LLM planning...")
    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Current state:\n{context}\n\nWhat should we do next?\n\nRespond with your decision in this format:\nACTION: [research|coding|end]\nREASONING: [your reasoning]\nINSTRUCTIONS: [specific instructions for next agent]")
    ]
    debug_print(f"Created {len(messages)} messages for planning")
    debug_print(f"Context length: {len(context)} characters")
    
    try:
        # Try structured output first
        debug_print("Attempting structured output...")
        structured_llm = llm.with_structured_output(OrchestratorPlan)
        plan = await structured_llm.ainvoke(messages)
        debug_print("Structured output successful")
        return plan
        
    except Exception as e:
        debug_print(f"Structured output failed: {str(e)[:200]}...")
        debug_print("Falling back to text parsing...")
        
        # Fallback to regular text completion
        try:
            response = await llm.ainvoke(messages)
            text = response.content if hasattr(response, 'content') else str(response)
            debug_print(f"Got text response: {text[:200]}...")
            
            # DEBUG: Print full response in debug mode
            if os.getenv("DEBUG_MODE"):
                debug_print("=== FULL LLM RESPONSE FOR DEBUGGING ===")
                debug_print(text)
                debug_print("=== END FULL RESPONSE ===")
            
            # Parse the response
            plan = parse_orchestrator_response(text)
            debug_print("Text parsing successful")
            return plan
            
        except Exception as parse_error:
            debug_print(f"Text parsing failed: {str(parse_error)}")
            # Ultimate fallback with simple logic
            return get_fallback_plan(context)


def parse_orchestrator_response(text: str) -> OrchestratorPlan:
    """
    Parse text response into OrchestratorPlan.
    
    Args:
        text: Raw text response from LLM
        
    Returns:
        OrchestratorPlan: Parsed plan
    """
    # Look for ACTION, REASONING, INSTRUCTIONS patterns
    action_match = re.search(r'ACTION:\s*(\w+)', text, re.IGNORECASE)
    reasoning_match = re.search(r'REASONING:\s*([^\n]+)', text, re.IGNORECASE)
    instructions_match = re.search(r'INSTRUCTIONS:\s*([^\n]+)', text, re.IGNORECASE)
    
    # Extract values with defaults
    action = action_match.group(1).lower() if action_match else "research"
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Default reasoning"
    instructions = instructions_match.group(1).strip() if instructions_match else "Continue with workflow"
    
    # Validate action
    if action not in ["research", "coding", "end"]:
        # Try to infer from text content
        if "research" in text.lower() or "search" in text.lower():
            action = "research"
        elif "code" in text.lower() or "implement" in text.lower():
            action = "coding"
        else:
            action = "end"
    
    return OrchestratorPlan(
        next_action=action,
        reasoning=reasoning,
        instructions=instructions
    )


def get_fallback_plan(context: str) -> OrchestratorPlan:
    """
    Ultimate fallback plan using simple logic.
    
    Args:
        context: Current workflow context
        
    Returns:
        OrchestratorPlan: Simple logic-based plan
    """
    debug_print("Using fallback planning logic")
    
    # Simple logic based on context
    if "Research Done: False" in context or "found 0 insights" in context:
        return OrchestratorPlan(
            next_action="research",
            reasoning="No research completed yet - need to search for Kaggle insights",
            instructions="Search for similar problems and extract relevant techniques"
        )
    elif "Code Generated: False" in context:
        return OrchestratorPlan(
            next_action="coding",
            reasoning="Research completed but no code generated yet",
            instructions="Generate and execute code based on the insights found"
        )
    else:
        return OrchestratorPlan(
            next_action="end",
            reasoning="Both research and coding appear to be completed",
            instructions="Workflow should be finalized"
        )


def create_orchestrator_node(callbacks=None, custom_model=None):
    """
    Creates the orchestrator agent node.
    Adapted from open_deep_research supervisor pattern.
    
    Args:
        callbacks: Optional callbacks for streaming
        custom_model: Dict with custom model config:
                     {"provider": "openai", "model": "gpt-4o"}
    
    Returns:
        Callable: Node function for LangGraph
    """
    async def orchestrator(state: DataScienceState) -> Command:
        """
        Orchestrator logic that determines workflow.
        
        Main flow:
        1. If no insights yet -> delegate to research
        2. If insights exist -> delegate to coding  
        3. If code executed -> evaluate and decide next
        4. If satisfied or max iterations -> end
        """
        print(f"[Orchestrator] Analyzing problem and planning next steps...")
        debug_print("Orchestrator agent started")
        debug_print(f"Input state keys: {list(state.keys())}")
        
        # Initialize LLM - use custom model if provided
        debug_print("Initializing LLM for orchestrator...")
        if custom_model:
            debug_print(f"Using custom model from parameters: {custom_model}")
            from llm_utils import create_custom_llm
            llm = create_custom_llm(
                provider=custom_model["provider"],
                model_id=custom_model["model"],
                temperature=0.0,
                streaming=True,
                callbacks=callbacks,
                agent_name="🧠 Orchestrator"
            )
        elif os.getenv("ORCHESTRATOR_PROVIDER") and os.getenv("ORCHESTRATOR_MODEL"):
            # Check for command-line environment variables
            provider = os.getenv("ORCHESTRATOR_PROVIDER")
            model = os.getenv("ORCHESTRATOR_MODEL")
            debug_print(f"Using custom model from environment: {provider}/{model}")
            from llm_utils import create_custom_llm
            llm = create_custom_llm(
                provider=provider,
                model_id=model,
                temperature=0.0,
                streaming=True,
                callbacks=callbacks,
                agent_name="🧠 Orchestrator"
            )
        else:
            debug_print("Using default LLM")
            llm = get_default_llm(temperature=0, streaming=True, callbacks=callbacks)
        debug_print("LLM initialized successfully")
        
        # Check current state and decide next action
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        debug_print(f"Current iteration: {current_iteration}/{max_iterations}")
        
        # Log current state details  
        research_done = len(state.get("search_queries", [])) > 0  # Research is done if queries were executed
        insights_found = len(state.get("kaggle_insights", [])) > 0
        code_generated = bool(state.get("generated_code", ""))
        code_executed = bool(state.get("execution_results", {}))
        
        debug_print(f"Research done: {research_done} (queries: {len(state.get('search_queries', []))}, insights: {len(state.get('kaggle_insights', []))})")
        debug_print(f"Code generated: {code_generated}")
        debug_print(f"Code executed: {code_executed}")
        debug_print(f"Errors: {len(state.get('error_messages', []))}")
        
        # Build context for decision
        context = f"""
        Problem: {state.get("problem_description", "No problem description")}
        Dataset Info: {state.get("dataset_info", {})}
        Current Iteration: {current_iteration}/{max_iterations}
        
        Research Done: {research_done} (Executed {len(state.get('search_queries', []))} searches, found {len(state.get('kaggle_insights', []))} insights)
        Code Generated: {code_generated}
        Code Executed: {code_executed}
        Errors: {state.get("error_messages", [])}
        """
        
        # Get the plan with fallback for models without structured output support
        debug_print("Getting planning decision...")
        plan = await get_orchestrator_plan(llm, context)
        debug_print(f"LLM returned plan: {plan}")
        
        print(f"[Orchestrator] Decision: {plan.next_action} - {plan.reasoning}")
        debug_print(f"Instructions: {plan.instructions}")
        
        # Update iteration count
        new_state = {"current_iteration": current_iteration + 1}
        
        # Safety check: prevent infinite loops
        if current_iteration >= max_iterations:
            print(f"[Orchestrator] → ENDING workflow: Maximum iterations ({max_iterations}) reached")
            debug_print("Max iterations reached, forcing end")
            final_summary = create_summary(state)
            return Command(
                goto=END,
                update={
                    **new_state,
                    "final_summary": final_summary,
                    "messages": state.get("messages", []) + [
                        AIMessage(content="Workflow ended after reaching maximum iterations.")
                    ],
                    # Pass through all important fields for final results
                    "generated_code": state.get("generated_code", ""),
                    "code_explanation": state.get("code_explanation", ""),
                    "created_scripts": state.get("created_scripts", []),
                    "execution_results": state.get("execution_results", {}),
                    "visualizations": state.get("visualizations", []),
                    "error_messages": state.get("error_messages", []),
                    "kaggle_insights": state.get("kaggle_insights", []),
                    "search_queries": state.get("search_queries", [])
                }
            )
        
        # Override LLM decision if needed to prevent loops
        if plan.next_action == "research" and research_done:
            print(f"[Orchestrator] ⚠️  LLM wants research but it's already done - routing to CODING instead")
            debug_print("Overriding LLM decision: research already done, going to coding")
            plan.next_action = "coding"
            plan.reasoning = "Research was already completed (queries executed), proceeding to code generation"
        
        # Route based on decision
        if plan.next_action == "research":
            # Send to research agent
            print(f"[Orchestrator] → Routing to RESEARCH agent: {plan.reasoning}")
            return Command(
                goto="research",
                update={
                    **new_state,
                    "messages": state.get("messages", []) + [
                        AIMessage(content=f"Delegating to research: {plan.instructions}")
                    ]
                }
            )
        elif plan.next_action == "coding":
            # Send to coding agent
            print(f"[Orchestrator] → Routing to CODING agent: {plan.reasoning}")
            return Command(
                goto="coding", 
                update={
                    **new_state,
                    "messages": state.get("messages", []) + [
                        AIMessage(content=f"Delegating to coding: {plan.instructions}")
                    ]
                }
            )
        else:
            # End workflow
            print(f"[Orchestrator] → ENDING workflow: {plan.reasoning}")
            debug_print("Creating final summary...")
            final_summary = create_summary(state)
            debug_print(f"Final summary created: {final_summary[:100]}...")
            return Command(
                goto=END,
                update={
                    **new_state,
                    "final_summary": final_summary,
                    "messages": state.get("messages", []) + [
                        AIMessage(content=final_summary)
                    ],
                    # Pass through all important fields for final results
                    "generated_code": state.get("generated_code", ""),
                    "code_explanation": state.get("code_explanation", ""),
                    "created_scripts": state.get("created_scripts", []),
                    "execution_results": state.get("execution_results", {}),
                    "visualizations": state.get("visualizations", []),
                    "error_messages": state.get("error_messages", []),
                    "kaggle_insights": state.get("kaggle_insights", []),
                    "search_queries": state.get("search_queries", [])
                }
            )
    
    return orchestrator

def analyze_problem(state: DataScienceState) -> Dict[str, Any]:
    """
    Analyzes the user's problem and dataset.
    Returns analysis for planning.
    """
    dataset_path = state.get("dataset_path")
    problem = state.get("problem_description", "")
    
    # Basic dataset analysis if path provided
    dataset_info = {}
    if dataset_path:
        try:
            df = pd.read_csv(dataset_path)
            dataset_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_columns": list(df.select_dtypes(include=['number']).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns)
            }
        except Exception as e:
            dataset_info = {"error": str(e)}
    
    return {
        "problem": problem,
        "dataset_info": dataset_info
    }

def evaluate_results(state: DataScienceState) -> bool:
    """
    Evaluates if the solution is satisfactory.
    Adapted from open_deep_research evaluation patterns.
    """
    # Check if we have results
    if not state.get("execution_results"):
        return False
    
    # Check for errors
    if state.get("error_messages"):
        # Minor errors might be okay, critical ones need fixing
        critical_errors = [e for e in state.get("error_messages", []) if "critical" in e.lower()]
        if critical_errors:
            return False
    
    # Check if we have visualizations or metrics
    has_visualizations = len(state.get("visualizations", [])) > 0
    has_metrics = bool(state.get("execution_results", {}).get("metrics"))
    
    # Simple heuristic: we need at least results and either viz or metrics
    return bool(state.get("execution_results")) and (has_visualizations or has_metrics)

def create_summary(state: DataScienceState) -> str:
    """
    Creates final solution summary.
    Combines insights, code, and results.
    """
    import json
    
    summary_parts = [
        "# 🔬 Data Science Solution Summary\n",
        f"## 📋 Problem Statement\n{state.get('problem_description', 'N/A')}\n",
    ]
    
    # Add dataset info
    if state.get("dataset_info"):
        info = state.get("dataset_info", {})
        summary_parts.append("## 📊 Dataset Overview")
        summary_parts.append(f"- **Shape**: {info.get('shape', 'N/A')}")
        summary_parts.append(f"- **File**: {info.get('filename', 'N/A')}")
        if info.get('columns'):
            summary_parts.append(f"- **Number of features**: {len(info['columns'])}")
        summary_parts.append("")
    
    # Add insights used
    if state.get("kaggle_insights"):
        summary_parts.append("## 🔍 Insights from Similar Kaggle Competitions")
        for i, insight in enumerate(state.get("kaggle_insights", [])[:5], 1):
            summary_parts.append(f"\n### {i}. {insight.get('title', 'Unknown Competition')}")
            # Only show competition name if it's not N/A
            if insight.get('competition_name') and insight.get('competition_name') != 'N/A':
                summary_parts.append(f"- **Competition**: {insight['competition_name']}")
            # Only show problem type if it's not N/A
            if insight.get('problem_type') and insight.get('problem_type') != 'N/A':
                summary_parts.append(f"- **Problem Type**: {insight['problem_type']}")
            summary_parts.append(f"- **Similarity Score**: {insight.get('score', 0):.3f}")
            if insight.get('ml_techniques'):
                summary_parts.append(f"- **Key Techniques**: {', '.join(insight['ml_techniques'][:5])}")
            if insight.get('key_insights'):
                summary_parts.append(f"- **Key Insights**: {insight['key_insights']}")
        summary_parts.append("")
    
    # Add approach and scripts created
    if state.get("code_explanation"):
        summary_parts.append("## 💡 Solution Approach")
        try:
            # Parse JSON if it's a string
            explanation = state.get("code_explanation")
            if isinstance(explanation, str):
                explanation = json.loads(explanation)
            
            if isinstance(explanation, dict):
                if explanation.get('overview'):
                    summary_parts.append(f"\n{explanation['overview']}")
                
                if explanation.get('key_steps'):
                    summary_parts.append("\n### Key Steps:")
                    for step in explanation['key_steps']:
                        summary_parts.append(f"- {step}")
                
                if explanation.get('techniques'):
                    summary_parts.append(f"\n### Techniques Used:")
                    for tech in explanation['techniques']:
                        summary_parts.append(f"- {tech}")
            else:
                summary_parts.append(f"\n{explanation}")
        except:
            # If parsing fails, just add as-is
            summary_parts.append(f"\n{state.get('code_explanation')}")
        summary_parts.append("")
    
    # Add created scripts
    if state.get("created_scripts"):
        summary_parts.append("## 📝 Scripts Created")
        for script in state.get("created_scripts", []):
            summary_parts.append(f"- `{script['name']}` - {script.get('path', '')}")
        summary_parts.append("")
    
    # Add results
    if state.get("execution_results"):
        results = state.get("execution_results", {})
        summary_parts.append("## 📈 Results")
        
        if results.get("success"):
            summary_parts.append("✅ **Code executed successfully!**")
        else:
            summary_parts.append("❌ **Code execution encountered errors**")
            if results.get("error"):
                summary_parts.append(f"\nError: {results['error']}")
        
        # Add execution output preview
        if results.get("output"):
            output_preview = results["output"][:500]
            summary_parts.append("\n### Execution Output (Preview):")
            summary_parts.append("```")
            summary_parts.append(output_preview)
            if len(results["output"]) > 500:
                summary_parts.append("... (output truncated)")
            summary_parts.append("```")
        
        # Add metrics if available
        if results.get("metrics"):
            summary_parts.append("\n### Performance Metrics:")
            for metric, value in results.get("metrics", {}).items():
                summary_parts.append(f"- **{metric}**: {value}")
        summary_parts.append("")
    
    # Extract performance metrics from output
    metrics_found = {}
    if state.get("execution_results", {}).get("output"):
        output = state.get("execution_results", {})["output"]
        lines = output.split('\n')
        
        # Look for common metric patterns
        import re
        for line in lines:
            # Look for accuracy (handle various formats)
            accuracy_match = re.search(r'accuracy[^:]*[:\s]+([0-9.]+)', line.lower())
            if accuracy_match:
                metrics_found['Accuracy'] = float(accuracy_match.group(1))
            
            # Look for precision/recall/f1
            for metric in ['precision', 'recall', 'f1-score', 'f1 score']:
                metric_match = re.search(rf'{metric}[:\s]+([0-9.]+)', line.lower())
                if metric_match:
                    metrics_found[metric.title().replace('-', ' ')] = float(metric_match.group(1))
            
            # Look for RMSE, MAE, MSE (use word boundaries to avoid MSE matching RMSE)
            for metric in ['rmse', 'mae', 'mse', 'r2', 'r-squared']:
                metric_match = re.search(rf'\b{metric}[:\s]+([0-9.]+)', line.lower())
                if metric_match:
                    metrics_found[metric.upper().replace('-', ' ')] = float(metric_match.group(1))
    
    # If we found metrics, add them to the results section
    if metrics_found:
        if not any("Performance Metrics:" in part for part in summary_parts):
            summary_parts.append("\n### 📊 Performance Metrics")
            for metric, value in metrics_found.items():
                # Format value appropriately
                if value < 1:  # Likely a percentage
                    summary_parts.append(f"- **{metric}**: {value:.4f} ({value*100:.2f}%)")
                else:
                    summary_parts.append(f"- **{metric}**: {value:.4f}")
            summary_parts.append("")
    
    # Add visualizations with embedded images
    # Check actual visualization files in both work directory and results directory
    from pathlib import Path
    
    viz_files = []
    
    # Try to find actual visualization files
    try:
        # First check work_dir if available
        if state.get("work_dir"):
            work_path = Path(state["work_dir"])
            # Look for all image files
            for pattern in ["*.png", "*.jpg", "*.jpeg"]:
                viz_files.extend(work_path.glob(pattern))
            
            # Also check results subdirectory
            results_path = work_path / "results"
            if results_path.exists():
                for pattern in ["*.png", "*.jpg", "*.jpeg"]:
                    viz_files.extend(results_path.glob(pattern))
        
        # Remove duplicates and sort
        viz_files = sorted(list(set(viz_files)))
        
        # Filter out non-visualization files (like logos, icons, etc.)
        viz_files = [f for f in viz_files if not any(skip in f.name.lower() for skip in ['logo', 'icon', 'favicon'])]
        
        debug_print(f"Found {len(viz_files)} visualization files")
    except Exception as e:
        debug_print(f"Error checking visualization files: {e}")
    
    # Add visualizations section if files found
    if viz_files:
        summary_parts.append("## 📊 Visualizations")
        summary_parts.append(f"\nGenerated {len(viz_files)} visualization(s):\n")
        
        # Group and describe visualizations based on filename
        for viz_file in viz_files:
            filename = viz_file.name.lower()
            
            # Determine visualization type and description
            if 'correlation' in filename or 'heatmap' in filename:
                title = "Correlation Heatmap"
                description = "Shows the correlation between different features in the dataset"
            elif 'decision_tree' in filename or 'tree' in filename:
                title = "Decision Tree Visualization"
                description = "Visual representation of the decision tree model structure"
            elif 'confusion' in filename or 'matrix' in filename:
                title = "Confusion Matrix"
                description = "Shows the classification performance across different classes"
            elif 'feature_importance' in filename or 'importance' in filename:
                title = "Feature Importance"
                description = "Displays the relative importance of each feature in the model"
            elif 'distribution' in filename or 'hist' in filename:
                title = "Data Distribution"
                description = "Shows the distribution of values in the dataset"
            elif 'scatter' in filename or 'plot' in filename:
                title = "Scatter Plot"
                description = "Visualizes the relationship between variables"
            elif 'roc' in filename or 'auc' in filename:
                title = "ROC Curve"
                description = "Receiver Operating Characteristic curve showing model performance"
            elif 'loss' in filename or 'learning' in filename:
                title = "Learning Curve"
                description = "Shows model performance over training iterations"
            else:
                # Generic title based on filename
                title = viz_file.stem.replace('_', ' ').replace('-', ' ').title()
                description = "Visualization generated by the analysis"
            
            summary_parts.append(f"### {title}")
            summary_parts.append(f"*{description}*\n")
            
            # For files in results directory, use relative path
            if 'results' in str(viz_file):
                summary_parts.append(f"![{title}]({viz_file.name})")
            else:
                # For files in work directory, need relative path from results
                try:
                    rel_path = f"../{viz_file.name}"
                    summary_parts.append(f"![{title}]({rel_path})")
                except:
                    summary_parts.append(f"![{title}]({viz_file.name})")
            
            summary_parts.append("")
    
    # Add key findings from execution output
    if state.get("execution_results", {}).get("output"):
        output = state.get("execution_results", {})["output"]
        
        # Try to extract key findings from output
        summary_parts.append("## 🔑 Key Findings")
        
        # Look for common patterns in output
        lines = output.split('\n')
        found_meaningful_findings = False
        key_findings = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            line_stripped = line.strip()
            
            # Skip warnings, errors, and file paths
            if any(skip in line_lower for skip in ['warning:', 'error:', 'futurewarning', 'deprecation', '/', '\\', '.py']):
                continue
                
            # Look for results patterns
            if any(keyword in line_lower for keyword in ['top ', 'ranking', 'most popular', 'highest', 'best', 'accuracy:', 'precision:', 'recall:', 'f1']):
                # Include this line
                key_findings.append(line_stripped)
                found_meaningful_findings = True
                
                # Add following lines that look like rankings (start with numbers or bullets)
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and (next_line[0].isdigit() or next_line.startswith('-') or next_line.startswith('•')):
                        key_findings.append(next_line)
                    elif not next_line:
                        continue
                    else:
                        break
            
            # Look for feature names, column lists, or important statistics
            elif 'columns:' in line_lower or 'features:' in line_lower or 'shape:' in line_lower:
                key_findings.append(line_stripped)
                found_meaningful_findings = True
            
            # Look for model results
            elif any(result in line_lower for result in ['trained', 'completed', 'saved', 'score:', 'metric:']):
                if len(line_stripped) > 10 and 'successfully' in line_lower:
                    key_findings.append(line_stripped)
                    found_meaningful_findings = True
        
        # Deduplicate and limit findings
        seen = set()
        unique_findings = []
        for finding in key_findings:
            if finding not in seen and len(finding) > 10:
                seen.add(finding)
                unique_findings.append(finding)
        
        if unique_findings:
            for finding in unique_findings[:8]:  # Limit to 8 key findings
                summary_parts.append(f"\n- {finding}")
        else:
            # If no meaningful findings, just note that the analysis completed
            summary_parts.append("\n- Analysis completed successfully")
            summary_parts.append("- Results saved to the work directory")
        
        summary_parts.append("")
    
    # Add final notes
    summary_parts.append("## 🎯 Summary")
    summary_parts.append(f"Successfully analyzed the problem '{state.get('problem_description', 'N/A')}' ")
    
    if state.get("kaggle_insights"):
        summary_parts.append(f"using insights from {len(state.get('kaggle_insights', []))} similar Kaggle competitions. ")
    
    if state.get("created_scripts"):
        num_scripts = len(state.get("created_scripts", []))
        summary_parts.append(f"Created {num_scripts} script(s) to solve the problem. ")
    
    if viz_files:
        summary_parts.append(f"Generated {len(viz_files)} visualization(s) to present the results.")
    
    summary_parts.append("\n\n---\n*Generated by Data Science Agent*")
    
    return "\n".join(summary_parts) 


# Standalone function for testing
async def analyze_problem_standalone(problem_description: str, dataset_info: Dict, llm) -> ProblemAnalysis:
    """
    Analyze a problem for testing purposes.
    
    Args:
        problem_description: Problem to analyze
        dataset_info: Dataset information
        llm: Language model
        
    Returns:
        ProblemAnalysis object
    """
    # Create prompt
    prompt = f"""
    Analyze this data science problem:
    
    Problem: {problem_description}
    Dataset: {dataset_info}
    
    Identify the problem type, data requirements, and suggested approaches.
    
    Respond in this format:
    PROBLEM_TYPE: [classification|regression|clustering|time_series|other]
    DATA_REQUIREMENTS:
    - [requirement 1]
    - [requirement 2]
    SUGGESTED_APPROACHES:
    - [approach 1]
    - [approach 2]
    """
    
    messages = [
        SystemMessage(content="You are a data science expert. Analyze the problem."),
        HumanMessage(content=prompt)
    ]
    
    try:
        # Try structured output first
        structured_llm = llm.with_structured_output(ProblemAnalysis)
        analysis = await structured_llm.ainvoke(messages)
        return analysis
    except Exception:
        # Fallback to simple parsing
        response = await llm.ainvoke(messages)
        text = response.content if hasattr(response, 'content') else str(response)
        
        # Simple parsing
        problem_type = "other"
        if re.search(r'PROBLEM_TYPE:\s*(\w+)', text, re.IGNORECASE):
            problem_type = re.search(r'PROBLEM_TYPE:\s*(\w+)', text, re.IGNORECASE).group(1)
        
        return ProblemAnalysis(
            problem_type=problem_type if problem_type in ["classification", "regression", "clustering", "time_series", "other"] else "other",
            data_requirements=["General data requirements"],
            suggested_approaches=["Standard machine learning approach"]
        ) 