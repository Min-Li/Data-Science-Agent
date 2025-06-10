"""
Improved Code Executor - Transparent execution without hidden manipulation.
"""

import subprocess
import sys
import os
import tempfile
import time
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool
from pydantic import Field
from work_dir_manager import get_work_dir_manager


class TransparentCodeExecutor(BaseTool):
    """
    Transparent code executor that runs code as-is without hidden manipulation.
    
    The coding agent is responsible for:
    - Loading datasets
    - Saving visualizations
    - Managing file outputs
    """
    
    name: str = "execute_python_code"
    description: str = """Execute Python code transparently without modification.
    
    The code should be complete and self-contained, including:
    - All necessary imports
    - Dataset loading
    - Result saving
    - Error handling
    """
    
    def _run(self, code: str) -> Dict[str, Any]:
        """Execute code without any hidden manipulation."""
        try:
            work_manager = get_work_dir_manager()
            
            # Save the code as-is
            if work_manager.work_dir:
                code_file = work_manager.save_file("generated_code.py", code)
                print(f"💾 Saved code to: {code_file}")
            
            # Get user approval if needed
            if os.getenv("AUTO_APPROVE_CODE") != "1":
                if not self._get_user_approval(code):
                    return {
                        "success": False,
                        "error": "Code execution cancelled by user",
                        "output": None
                    }
            
            print("\n🚀 Executing code...")
            
            # Create execution file
            if work_manager.work_dir:
                work_dir_abs = Path(work_manager.work_dir).resolve()
                exec_file = work_dir_abs / "execution_script.py"
                with open(exec_file, 'w') as f:
                    f.write(code)
                execution_cwd = str(work_dir_abs)
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    exec_file = f.name
                execution_cwd = os.getcwd()
            
            try:
                # Execute with proper environment
                env = os.environ.copy()
                
                # Set PYTHONPATH for imports
                project_src = Path(__file__).parent.parent.resolve()
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{project_src}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = str(project_src)
                
                # Run the code
                process = subprocess.Popen(
                    [sys.executable, str(exec_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=execution_cwd,
                    env=env,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Capture output
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    line = line.rstrip('\n')
                    if line:
                        print(f"🐍 {line}")
                        output_lines.append(line)
                
                process.wait()
                output = '\n'.join(output_lines)
                
                if process.returncode != 0:
                    error_msg = f"Process exited with code {process.returncode}"
                    return {
                        "success": False,
                        "error": error_msg,
                        "output": output  # Include output for debugging
                    }
                else:
                    return {
                        "success": True,
                        "output": output,
                        "error": None
                    }
                    
            finally:
                # Clean up temp file if used
                if not work_manager.work_dir:
                    try:
                        os.unlink(exec_file)
                    except:
                        pass
                        
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "output": None
            }
    
    def _get_user_approval(self, code: str) -> bool:
        """Get user approval before executing code."""
        print("\n" + "="*80)
        print("🚨 CODE EXECUTION APPROVAL REQUIRED")
        print("="*80)
        print("The following code will be executed:")
        print("-"*80)
        print(code)
        print("-"*80)
        print("✅ Type 'run' or press Enter to execute")
        print("❌ Type 'no' or 'cancel' to abort")
        print("="*80)
        
        try:
            response = input("Your decision [run/cancel]: ").strip().lower()
            return response in ['', 'run', 'yes', 'y', 'execute', 'ok']
        except KeyboardInterrupt:
            print("\n❌ Execution cancelled by user")
            return False
    
    async def _arun(self, code: str) -> Dict[str, Any]:
        """Async version - just calls sync version."""
        return self._run(code)


# Example of how the improved prompts would look:
IMPROVED_CODE_GENERATION_TEMPLATE = """
Generate a complete Python script to solve this data science problem.

Problem: {problem}

Work Directory Structure:
- Current directory: {work_dir}
- Dataset location: {dataset_path}
- Save outputs to: {results_dir}

Requirements:
1. Load the dataset from the specified path
2. Perform the requested analysis
3. Save all visualizations to the results directory with descriptive names
4. Save any models, predictions, or metrics to the results directory
5. Print informative summaries to stdout

Example code structure:
```python
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set up paths
results_dir = Path("{results_dir}")
results_dir.mkdir(exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv("{dataset_path}")
print(f"Dataset shape: {{df.shape}}")

# Your analysis here...

# Save visualizations with descriptive names
plt.figure(figsize=(10, 6))
# ... create visualization ...
plt.savefig(results_dir / "analysis_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved visualization: analysis_results.png")

# Save metrics
metrics = {{"accuracy": 0.95, "f1_score": 0.93}}
with open(results_dir / "model_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)
print("✅ Saved metrics: model_metrics.json")
```

Generate complete, self-contained code that handles all aspects of the solution.
""" 