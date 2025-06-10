"""
Utility Functions
=================

Helper functions adapted from open_deep_research/utils.py.
Provides common functionality for data processing, formatting, and visualization.

Main utilities:
- Dataset analysis and statistics
- Result formatting and display
- Visualization helpers
- File I/O operations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import json
import base64
from io import BytesIO

def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """
    Analyzes uploaded dataset and returns statistics.
    
    Returns:
        dict: Dataset information including shape, columns, types, missing values
    """
    pass

def format_kaggle_insights(insights: List[Dict[str, str]]) -> str:
    """
    Formats Kaggle competition insights for display.
    Adapted from open_deep_research formatting utilities.
    
    Args:
        insights: List of competition solutions
        
    Returns:
        str: Formatted markdown string
    """
    pass

def encode_plot_to_base64(fig) -> str:
    """
    Encodes matplotlib/plotly figure to base64 for web display.
    Used by Streamlit interface.
    
    Args:
        fig: Matplotlib or Plotly figure object
        
    Returns:
        str: Base64 encoded image string
    """
    pass

def parse_code_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses output from Riza code execution.
    Extracts results, plots, and metrics.
    
    Args:
        output: Raw output from code executor
        
    Returns:
        dict: Structured results with visualizations
    """
    pass

def create_error_summary(errors: List[str]) -> str:
    """
    Creates user-friendly error summary.
    Adapted from open_deep_research error handling.
    
    Args:
        errors: List of error messages
        
    Returns:
        str: Formatted error summary
    """
    pass

def load_vector_database(db_path: str) -> Any:
    """
    Loads pre-built vector database from disk.
    Uses numpy arrays for embeddings.
    
    Args:
        db_path: Path to vector database directory
        
    Returns:
        InMemoryVectorStore: Loaded vector store
    """
    pass

def format_final_report(state: 'DataScienceState') -> str:
    """
    Creates final solution report.
    Adapted from open_deep_research report generation.
    
    Args:
        state: Final agent state
        
    Returns:
        str: Markdown formatted report
    """
    pass 