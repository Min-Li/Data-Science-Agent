# Quick fix: Make visualization saving optional

def _prepare_code(self, code: str, auto_save_viz: bool = False) -> str:
    """Prepare code with optional visualization saving."""
    
    header = '''# =============================================================================
# AUTO-GENERATED DATA SCIENCE CODE
# =============================================================================
'''
    
    # Check if code already handles visualization saving
    has_savefig = 'savefig' in code
    has_results_dir = 'results' in code or 'results_dir' in code
    
    # Only add visualization saving if:
    # 1. auto_save_viz is True
    # 2. Code doesn't already handle saving
    if auto_save_viz and not (has_savefig and has_results_dir):
        # Add the visualization saving code
        return f"{header}\n{code}\n{self.viz_saving_code}"
    else:
        # Just return the code as-is
        return f"{header}\n{code}" 