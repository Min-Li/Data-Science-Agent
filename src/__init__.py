"""
Data Science Agent - Automated solution generation using Kaggle insights.
"""

from state import DataScienceState
from graph import (
    create_data_science_graph,
    run_data_science_agent,
    solve_kaggle_problem,
    analyze_dataset
)

__version__ = "0.1.0"

__all__ = [
    "DataScienceState",
    "create_data_science_graph", 
    "run_data_science_agent",
    "solve_kaggle_problem",
    "analyze_dataset"
] 