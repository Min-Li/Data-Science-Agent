"""
State Management - Shared State Between Agents
=============================================

This module defines the shared state structure that flows between all agents in the system.
Think of it as the "memory" that agents use to communicate with each other.

Core Concept:
------------
LangGraph requires a well-defined state schema that all nodes (agents) can read from
and write to. This is our contract between agents - what data they can expect and produce.

DataScienceState Fields:
-----------------------
- **problem_description**: The user's original problem statement
- **dataset_info**: Metadata about the dataset (shape, columns, path)
- **dataset_path**: Actual file path to the dataset
- **search_queries**: Queries generated for Kaggle search
- **kaggle_insights**: Results from searching similar Kaggle competitions
- **generated_code**: Python code produced by the coding agent
- **code_explanation**: Natural language explanation of the code
- **execution_results**: Output from running the generated code
- **error_messages**: Any errors encountered during execution
- **visualizations**: List of generated plots/charts (as base64 or file paths)
- **final_summary**: Complete solution summary with metrics and insights
- **work_dir**: Working directory for saving files and results
- **current_iteration**: Tracks workflow progress (prevents infinite loops)

State Flow Example:
------------------
1. User Input → Initial State (problem + dataset)
2. Orchestrator → Adds planning decisions
3. Research Agent → Adds kaggle_insights
4. Coding Agent → Adds generated_code + execution_results
5. Orchestrator → Evaluates and adds final_summary

Interview Notes:
---------------
- This is a Pydantic BaseModel for type safety and validation
- The state accumulates data - agents ADD to it, not replace
- Uses MessagesState mixin for LangChain message compatibility
- The 'reducer' pattern allows merging updates (e.g., appending to lists)
- Critical for debugging: all agent outputs are preserved here
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class DataScienceState(MessagesState):
    """
    Main state object for the data science agent workflow.
    Inherits from MessagesState to track conversation history.
    
    Adapted from open_deep_research ResearchState pattern.
    """
    # Problem definition
    problem_description: str = Field(default="", description="User's problem statement")
    dataset_path: Optional[str] = Field(default=None, description="Path to uploaded dataset")
    dataset_info: Dict[str, Any] = Field(default_factory=dict, description="Dataset statistics")
    
    # Research phase  
    kaggle_insights: List[Dict[str, str]] = Field(
        default_factory=list, 
        description="Retrieved Kaggle competition solutions"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Vector search queries used"
    )
    
    # Coding phase
    generated_code: str = Field(default="", description="Generated Python solution")
    code_explanation: str = Field(default="", description="Explanation of approach")
    created_scripts: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of scripts created: [{'name': 'script.py', 'path': '/path/to/script.py'}]"
    )
    script_outputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Outputs from each script execution"
    )
    
    # Execution phase
    execution_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from code execution"
    )
    error_messages: List[str] = Field(
        default_factory=list,
        description="Any errors encountered"
    )
    
    # Outputs
    visualizations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Generated plots and charts"
    )
    final_summary: str = Field(default="", description="Summary of solution")
    
    # Workflow control (from open_deep_research)
    max_iterations: int = Field(default=20, description="Max agent iterations")
    current_iteration: int = Field(default=0, description="Current iteration count")
    
    # Work directory information
    work_dir: Optional[str] = Field(default=None, description="Work directory path for saving files") 