"""
Data Science Agent Graph - Multi-Agent Workflow Orchestration
============================================================

This module implements the core LangGraph workflow that coordinates multiple AI agents
to solve data science problems. It's the central nervous system of the application.

Key Components:
--------------
1. **StateGraph**: LangGraph's state machine that manages agent transitions
2. **DataScienceState**: Shared state object passed between agents
3. **Agent Nodes**: Research, Coding, and Orchestrator agents
4. **Routing Logic**: Determines which agent to invoke next

Workflow Flow:
-------------
1. User provides problem description and dataset
2. Orchestrator decides: Do we need research or can we code directly?
3. Research Agent searches Kaggle for similar problems (if needed)
4. Coding Agent generates and executes Python solution
5. Orchestrator evaluates results and decides to iterate or finish

Key Features:
------------
- **Conditional Routing**: Smart decisions on agent transitions
- **Error Recovery**: Automatic retry on code execution failures
- **State Persistence**: All agent outputs stored in shared state
- **Custom Model Support**: Each agent can use different LLM providers

Usage:
-----
```python
graph = create_data_science_graph(agent_models={
    "orchestrator": {"provider": "openai", "model": "gpt-4o"},
    "research": {"provider": "anthropic", "model": "claude-sonnet"},
    "coding": {"provider": "openai", "model": "gpt-4o"}
})

async for output in graph.astream(initial_state, config):
    # Process agent outputs
```

Interview Notes:
---------------
- This is the MAIN WORKFLOW file - everything flows through here
- Uses LangGraph for state management (alternative to LangChain's AgentExecutor)
- Implements conditional edges for dynamic routing between agents
- Each agent is a separate node that updates the shared state
- The orchestrator acts as the "brain" deciding next steps
"""

from typing import Dict, Any, Literal
import asyncio
import os
import logging
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import DataScienceState
from agents.orchestrator import create_orchestrator_node
from agents.research_agent import create_research_node
from agents.coding_agent import create_coding_node

# Set up logger
logger = logging.getLogger(__name__)

def debug_print(message: str):
    """Debug print that only shows in debug mode."""
    if os.getenv("DEBUG_MODE"):
        print(f"🐛 [GRAPH] {message}")
    logger.debug(f"[GRAPH] {message}")


def create_data_science_graph(callbacks=None, agent_models=None):
    """
    Create the main data science workflow graph.
    
    The graph follows this pattern:
    1. Orchestrator analyzes the problem
    2. Based on state, routes to:
       - Research (if no insights)
       - Coding (if insights exist)
       - End (if solution is satisfactory or max iterations)
    3. After each step, returns to orchestrator for next decision
    
    Args:
        callbacks: Optional callbacks for streaming. Can be:
                  - A single callback list (used for all agents)
                  - A dict with agent names as keys and callback lists as values
        agent_models: Dict with model configurations for each agent:
                     {
                         "orchestrator": {"provider": "openai", "model": "gpt-4o"},
                         "research": {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
                         "coding": {"provider": "openai", "model": "gpt-4o"}
                     }
    
    Returns:
        Compiled StateGraph
    """
    debug_print("Creating data science workflow graph...")
    
    # Create the graph
    workflow = StateGraph(DataScienceState)
    debug_print("StateGraph instance created")
    
    # Handle different callback formats
    debug_print(f"Processing callbacks: {type(callbacks)}")
    if isinstance(callbacks, dict):
        orchestrator_callbacks = callbacks.get("orchestrator", None)
        research_callbacks = callbacks.get("research", None)
        coding_callbacks = callbacks.get("coding", None)
        debug_print("Using agent-specific callbacks")
    else:
        # Use same callbacks for all agents
        orchestrator_callbacks = callbacks
        research_callbacks = callbacks
        coding_callbacks = callbacks
        debug_print("Using shared callbacks for all agents")
    
    # Get model configurations
    debug_print(f"Processing agent models: {agent_models}")
    if agent_models:
        orchestrator_model = agent_models.get("orchestrator")
        research_model = agent_models.get("research")
        coding_model = agent_models.get("coding")
        debug_print("Using custom models for agents")
    else:
        orchestrator_model = None
        research_model = None
        coding_model = None
        debug_print("Using default models for all agents")
    
    # Add nodes with custom models
    debug_print("Creating orchestrator node...")
    workflow.add_node("orchestrator", create_orchestrator_node(
        callbacks=orchestrator_callbacks,
        custom_model=orchestrator_model
    ))
    debug_print("Creating research node...")
    workflow.add_node("research", create_research_node(
        callbacks=research_callbacks,
        custom_model=research_model
    ))
    debug_print("Creating coding node...")
    workflow.add_node("coding", create_coding_node(
        callbacks=coding_callbacks,
        custom_model=coding_model
    ))
    debug_print("All nodes added to workflow")
    
    # Add edges
    debug_print("Setting up workflow edges...")
    # Start with orchestrator
    workflow.set_entry_point("orchestrator")
    debug_print("Set orchestrator as entry point")
    
    # The orchestrator uses Command to route, so we don't need conditional edges
    # Just add the normal edges for when agents complete
    workflow.add_edge("research", "orchestrator")
    workflow.add_edge("coding", "orchestrator")
    debug_print("Added edges: research->orchestrator, coding->orchestrator")
    
    # Compile the graph
    debug_print("Compiling workflow graph...")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    debug_print("Workflow graph compiled successfully")
    
    return app


async def run_data_science_agent(
    problem_description: str,
    dataset_info: Dict[str, Any],
    max_iterations: int = 3,
    thread_id: str = "default",
    llm_provider: str = None,
    llm_model: str = None,
    callbacks=None
) -> Dict[str, Any]:
    """
    Run the complete data science agent workflow.
    
    Args:
        problem_description: Description of the problem to solve
        dataset_info: Information about the dataset (path, description, etc.)
        max_iterations: Maximum number of research/coding iterations
        thread_id: Thread ID for conversation memory
        llm_provider: LLM provider to use (openai, anthropic, gemini, deepseek)
        llm_model: Specific model to use (optional)
        
    Returns:
        Final state with solution and results
    """
    debug_print(f"Starting data science agent workflow")
    debug_print(f"Problem: {problem_description[:100]}...")
    debug_print(f"Dataset info: {dataset_info}")
    debug_print(f"Max iterations: {max_iterations}")
    debug_print(f"LLM provider: {llm_provider}, model: {llm_model}")
    
    # Set LLM configuration if provided
    if llm_provider:
        import os
        os.environ["LLM_PROVIDER"] = llm_provider
        debug_print(f"Set LLM_PROVIDER to {llm_provider}")
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model
        debug_print(f"Set LLM_MODEL to {llm_model}")
    
    # Create agent models configuration if provider/model specified
    agent_models = None
    if llm_provider or llm_model:
        # Use the same model for all agents
        model_config = {}
        if llm_provider:
            model_config["provider"] = llm_provider
        if llm_model:
            model_config["model"] = llm_model
        
        agent_models = {
            "orchestrator": model_config,
            "research": model_config,
            "coding": model_config
        }
        debug_print(f"Created agent_models configuration: {agent_models}")
    
    # Create the graph
    debug_print("Creating workflow graph...")
    app = create_data_science_graph(callbacks, agent_models)
    
    # Initialize state
    debug_print("Initializing agent state...")
    
    # Extract work_dir from dataset_info if available
    work_dir = None
    if isinstance(dataset_info, dict) and "work_dir" in dataset_info:
        work_dir = dataset_info["work_dir"]
        # Remove work_dir from dataset_info to avoid confusion
        dataset_info_clean = {k: v for k, v in dataset_info.items() if k != "work_dir"}
    else:
        dataset_info_clean = dataset_info
    
    initial_state = {
        "problem_description": problem_description,
        "dataset_info": dataset_info_clean,
        "dataset_path": dataset_info.get("path") if isinstance(dataset_info, dict) else None,
        "max_iterations": max_iterations,
        "current_iteration": 0,
        "search_queries": [],
        "kaggle_insights": [],
        "generated_code": "",
        "code_explanation": "",
        "execution_results": {},
        "error_messages": [],
        "visualizations": [],
        "final_summary": "",
        "messages": [],  # Required by MessagesState
        "work_dir": work_dir  # Set from dataset_info or None
    }
    debug_print(f"Initial state created with {len(initial_state)} fields, work_dir: {work_dir}")
    
    # Configuration for the run
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50  # Increase limit but with safety checks in orchestrator
    }
    debug_print(f"Running with thread_id: {thread_id}, recursion_limit: 50")
    
    # Run the graph
    debug_print("Starting workflow execution...")
    final_state = None
    iteration_count = 0
    async for output in app.astream(initial_state, config):
        iteration_count += 1
        debug_print(f"Workflow iteration {iteration_count}: received output")
        
        # Print progress
        for node, state in output.items():
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] {node.upper()} completed")
            debug_print(f"Node '{node}' completed with state keys: {list(state.keys()) if state else 'None'}")
            
            # Print some details about the state
            if state:
                if node == "research" and state.get("kaggle_insights"):
                    print(f"  → Found {len(state['kaggle_insights'])} relevant Kaggle competitions")
                    debug_print(f"Research insights: {[insight.get('title', 'Unknown') for insight in state['kaggle_insights'][:3]]}")
                elif node == "coding" and state.get("generated_code"):
                    print(f"  → Generated {len(state['generated_code'])} characters of code")
                    debug_print(f"Code preview: {state['generated_code'][:200]}...")
                    if state.get("execution_results", {}).get("success"):
                        print(f"  → Code executed successfully")
                        debug_print(f"Execution output: {state.get('execution_results', {}).get('output', '')[:200]}...")
                elif node == "orchestrator":
                    print(f"  → Iteration {state.get('current_iteration', 0)}/{max_iterations}")
                    debug_print(f"Orchestrator planning completed")
            
            # Store the latest state
            if state:
                final_state = state
    
    debug_print(f"Workflow completed after {iteration_count} iterations")
    debug_print(f"Final state contains: {list(final_state.keys()) if final_state else 'None'}")
    return final_state


def create_simple_graph():
    """
    Create a simplified version of the graph for testing.
    
    This version:
    - Always does research first
    - Then coding
    - Then ends
    
    Useful for initial testing without complex routing.
    """
    workflow = StateGraph(DataScienceState)
    
    # Add nodes
    workflow.add_node("orchestrator", create_orchestrator_node())
    workflow.add_node("research", create_research_node())  
    workflow.add_node("coding", create_coding_node())
    
    # Simple linear flow
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "research")
    workflow.add_edge("research", "coding")
    workflow.add_edge("coding", END)
    
    # Compile without memory for simplicity
    app = workflow.compile()
    
    return app


# Convenience functions for common use cases
async def solve_kaggle_problem(
    problem_description: str,
    dataset_path: str,
    competition_context: str = ""
) -> Dict[str, Any]:
    """
    Solve a Kaggle-style problem with the agent.
    
    Args:
        problem_description: What to predict/classify/analyze
        dataset_path: Path to the dataset file
        competition_context: Additional context about the competition
    
    Returns:
        Solution with code and results
    """
    # Format dataset info
    dataset_info = {
        "path": dataset_path,
        "description": f"Dataset from {dataset_path}",
        "context": competition_context
    }
    
    # Add competition context to problem if provided
    if competition_context:
        problem_description = f"{problem_description}\n\nCompetition context: {competition_context}"
    
    # Run the agent
    result = await run_data_science_agent(
        problem_description=problem_description,
        dataset_info=dataset_info,
        max_iterations=3
    )
    
    return result


async def analyze_dataset(
    dataset_path: str,
    analysis_goal: str = "Perform exploratory data analysis"
) -> Dict[str, Any]:
    """
    Analyze a dataset with the agent.
    
    Args:
        dataset_path: Path to the dataset
        analysis_goal: What kind of analysis to perform
        
    Returns:
        Analysis results with visualizations
    """
    dataset_info = {
        "path": dataset_path,
        "description": f"Dataset for analysis"
    }
    
    result = await run_data_science_agent(
        problem_description=analysis_goal,
        dataset_info=dataset_info,
        max_iterations=2  # Usually don't need many iterations for EDA
    )
    
    return result


# For testing individual components
async def test_research_only(problem_description: str) -> Dict[str, Any]:
    """Test just the research component."""
    from agents.research_agent import run_research_agent
    
    insights = await run_research_agent(
        problem_description=problem_description,
        dataset_info="Test dataset"
    )
    
    return insights


async def test_coding_only(
    problem_description: str,
    kaggle_insights: list,
    dataset_path: str = None
) -> Dict[str, Any]:
    """Test just the coding component."""
    from agents.coding_agent import run_coding_agent
    
    result = await run_coding_agent(
        problem_description=problem_description,
        dataset_info=f"Dataset at {dataset_path}" if dataset_path else "Test dataset",
        kaggle_insights=kaggle_insights,
        dataset_path=dataset_path
    )
    
    return result


if __name__ == "__main__":
    # Example usage
    async def main():
        # Test the complete workflow
        result = await run_data_science_agent(
            problem_description="Build a model to predict house prices based on features like size, location, and amenities.",
            dataset_info={
                "path": "data/house_prices.csv",
                "description": "House prices dataset with 80 features"
            }
        )
        
        print("\n=== Final Results ===")
        print(f"Solution found: {result.get('code') is not None}")
        print(f"Execution successful: {result.get('execution_error') is None}")
        if result.get('final_summary'):
            print(f"\nSummary: {result['final_summary']}")
    
    # Run the example
    asyncio.run(main()) 