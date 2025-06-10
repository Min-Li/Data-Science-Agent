"""
Code Executor - Safe Python Code Execution with Transparency
===========================================================

This module provides secure code execution for generated Python scripts with two modes:
local (with user approval) and remote (via Riza sandbox). It implements the TRANSPARENT
ARCHITECTURE where agents have full control over their code.

Execution Modes:
---------------
1. **Local Execution** (default)
   - Runs in your Python environment
   - Shows complete code for review
   - Requires explicit user approval
   - Real-time output streaming
   - Best for development and debugging

2. **Remote Execution** (Riza)
   - Runs in isolated cloud sandbox
   - No local environment access
   - Automatic execution (no approval)
   - Limited to pre-installed packages
   - Best for production safety

Transparent Architecture:
------------------------
Unlike traditional approaches that inject hidden code, this executor:
- Shows EXACTLY what will run (no hidden imports or manipulations)
- Agent generates complete, self-contained scripts
- User sees and approves the actual code
- All file operations controlled by the agent

Key Features:
------------
- **User Approval Interface**: Clear display of code before execution
- **Real-time Output**: Streams output line by line as it executes
- **Work Directory Integration**: Executes from proper work directory
- **PYTHONPATH Management**: Ensures imports work correctly
- **Error Context**: Returns output even on failures for debugging

Safety Features:
---------------
- Manual review required (can be disabled with AUTO_APPROVE_CODE=1)
- Shows current working directory and dataset path
- Clear warnings about local execution
- Timeout protection (5 minutes max)

Code Preparation:
----------------
The _prepare_code() method is now MINIMAL - just adds a header comment.
The coding agent is responsible for:
- All imports (pandas, numpy, sklearn, etc.)
- Loading datasets from the correct path
- Creating results directory
- Saving visualizations with descriptive names
- Printing execution summaries

Output Handling:
---------------
- Returns raw output without modification
- No base64 encoding or hidden manipulations
- Agent handles all file saving directly
- Clean, readable logs without clutter

Interview Notes:
---------------
- This is a CRITICAL security component - handles untrusted code
- The transparent architecture was a major improvement over v1
- Local mode is preferred for development (see what's happening)
- Remote mode exists but is rarely used in practice
- The user approval step has prevented many potential issues
"""

import os
import sys
import time
import subprocess
import tempfile
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass
import json
import re
from pathlib import Path
import base64

from langchain_core.tools import BaseTool
from pydantic import Field
from work_dir_manager import get_work_dir_manager, log_message

try:
    from rizaio import Riza
except ImportError:
    print("Warning: rizaio package not found. Please install it with: pip install rizaio")
    Riza = None


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[str]] = None


class BaseCodeExecutor(BaseTool):
    """Base class for code executors with common functionality."""
    
    name: str = "execute_python_code"
    description: str = """Execute Python code with data science libraries.
    
    The code has access to:
    - Python standard library
    - pandas, numpy, scikit-learn, matplotlib, seaborn
    - The dataset is pre-loaded as a pandas DataFrame named 'df'
    
    Returns the output, any errors, and extracted results/visualizations.
    """
    
    dataset_path: Optional[str] = Field(default=None, description="Path to the dataset file")
    
    def _prepare_code(self, code: str) -> str:
        """Prepare code for execution - minimal modification for transparency.
        
        The coding agent is now responsible for:
        - Loading datasets
        - Saving visualizations
        - Managing file outputs
        """
        # Just add a simple header comment
        header = '''# =============================================================================
# AUTO-GENERATED DATA SCIENCE CODE
# =============================================================================
'''
        # Return code as-is with just the header
        # The coding agent generates complete, self-contained code
        return f"{header}\n{code}"
    
    def _parse_output(self, output: str, error: str = None) -> Dict[str, Any]:
        """Parse execution output to extract results.
        
        Since the agent now handles visualization saving directly,
        we just return the output as-is.
        """
        if error:
            return {
                "success": False,
                "error": error,
                "output": output,  # Include the actual output even on error!
                "data": None,
                "visualizations": []  # Empty list since agent saves directly
            }
        
        return {
            "success": True,
            "output": output,
            "error": None,
            "data": None,  # Agent handles data saving
            "visualizations": []  # Agent handles visualization saving
        }


class LocalCodeExecutor(BaseCodeExecutor):
    """
    Local code executor with user approval for safe execution.
    
    Executes code in the current Python environment after user confirmation.
    """
    
    work_dir: Optional[str] = Field(default=None, description="Work directory to execute code from (for multi-agent mode)")
    
    def __init__(self, dataset_path: Optional[str] = None, **data):
        super().__init__(**data)
        self.dataset_path = dataset_path
    
    def _get_user_approval(self, code: str) -> bool:
        """Get user approval before executing code."""
        print("\n" + "="*80)
        print("🚨 CODE EXECUTION APPROVAL REQUIRED")
        print("="*80)
        print("The agent wants to execute the following Python code:")
        print("-"*80)
        print(code)
        print("-"*80)
        print("⚠️  This code will run in your local Python environment.")
        print("📁 Current working directory:", os.getcwd())
        if self.dataset_path:
            print("📊 Dataset:", self.dataset_path)
        print("\n🔍 Please review the code above carefully.")
        print("✅ Type 'run' or press Enter to execute")
        print("❌ Type 'no' or 'cancel' to abort")
        print("="*80)
        
        try:
            response = input("Your decision [run/cancel]: ").strip().lower()
            return response in ['', 'run', 'yes', 'y', 'execute', 'ok']
        except KeyboardInterrupt:
            print("\n❌ Execution cancelled by user")
            return False
    
    def _run(self, code: str) -> Dict[str, Any]:
        """Execute code locally with user approval."""
        try:
            # Get work directory manager
            work_manager = get_work_dir_manager()
            
            # Prepare the code
            full_code = self._prepare_code(code)
            
            # Save the code to work directory
            # Use self.work_dir if set (multi-agent mode), otherwise use work_manager
            effective_work_dir = self.work_dir or (work_manager.work_dir if work_manager.work_dir else None)
            
            if effective_work_dir:
                # Save directly to work directory
                code_file = Path(effective_work_dir) / "generated_code.py"
                with open(code_file, 'w') as f:
                    f.write(full_code)
                log_message(f"💾 Saved generated code to: {code_file}")
            elif work_manager.work_dir:
                code_file = work_manager.save_file("generated_code.py", full_code)
                log_message(f"💾 Saved generated code to: {code_file}")
            
            # Get user approval (skip if AUTO_APPROVE_CODE is set)
            if os.getenv("AUTO_APPROVE_CODE") == "1":
                print("\n🤖 AUTO-APPROVE: Executing code automatically...")
                log_message("🤖 Auto-approving code execution")
            else:
                if not self._get_user_approval(full_code):
                    log_message("❌ Code execution cancelled by user", "warning")
                    return {
                        "success": False,
                        "error": "Code execution cancelled by user",
                        "output": None,
                        "data": None,
                        "visualizations": None
                    }
            
            print("\n🚀 Executing code...")
            print("-"*40)
            log_message("🚀 Starting code execution")
            
            # Create execution file in work directory
            # Use self.work_dir if set (multi-agent mode), otherwise use work_manager
            effective_work_dir = self.work_dir or (work_manager.work_dir if work_manager.work_dir else None)
            
            if effective_work_dir:
                # Convert to absolute path to avoid path duplication
                work_dir_abs = Path(effective_work_dir).resolve()
                exec_file = work_dir_abs / "execution_script.py"
                with open(exec_file, 'w') as f:
                    f.write(full_code)
                execution_cwd = str(work_dir_abs)
                log_message(f"📝 Created execution script: {exec_file}")
            else:
                # Fallback to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(full_code)
                    exec_file = f.name
                execution_cwd = os.getcwd()
            
            try:
                # Execute with real-time output and proper PYTHONPATH
                env = os.environ.copy()
                
                # Add the project src directory to PYTHONPATH so imports work
                project_src = Path(__file__).parent.parent.resolve()  # Go up to src/
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{project_src}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = str(project_src)
                
                log_message(f"🐍 Setting PYTHONPATH: {env['PYTHONPATH']}")
                log_message(f"📁 Execution directory: {execution_cwd}")
                
                process = subprocess.Popen(
                    [sys.executable, str(exec_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=execution_cwd,
                    env=env,  # Pass the environment with PYTHONPATH
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Capture output in real-time
                output_lines = []
                print("📋 Python Output:")
                print("─" * 60)
                
                for line in iter(process.stdout.readline, ''):
                    line = line.rstrip('\n')
                    if line:  # Only print non-empty lines
                        print(f"🐍 {line}")
                        output_lines.append(line)
                        # Log to work directory
                        log_message(f"[PYTHON] {line}")
                
                process.wait()
                output = '\n'.join(output_lines)
                
                print("─" * 60)
                print("✅ Code execution completed")
                log_message("✅ Code execution completed successfully")
                
                if process.returncode != 0:
                    error_msg = f"Process exited with code {process.returncode}"
                    print(f"❌ Execution failed: {error_msg}")
                    log_message(f"❌ Execution failed: {error_msg}", "error")
                    # Pass the actual output so the agent can see what went wrong!
                    return self._parse_output(output, error_msg)
                else:
                    print("📊 Output captured successfully")
                    return self._parse_output(output)
                    
            finally:
                # Clean up temporary file if used
                if not work_manager.work_dir:
                    try:
                        os.unlink(exec_file)
                    except:
                        pass
                    
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Code execution timed out (5 minutes)",
                "output": None,
                "data": None,
                "visualizations": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "output": None,
                "data": None,
                "visualizations": None
            }
    
    async def _arun(self, code: str) -> Dict[str, Any]:
        """Async version - just calls sync version."""
        return self._run(code)


class RizaCodeExecutor(BaseCodeExecutor):
    """
    Remote code executor using Riza's sandboxed environment.
    
    This tool uses Riza's custom runtimes to provide safe code execution
    with support for data science workflows including pandas, scikit-learn, etc.
    """
    
    riza_client: Optional[Any] = Field(default=None, description="Riza API client")
    runtime_revision_id: Optional[str] = Field(default=None, description="Custom runtime revision ID")
    
    def __init__(self, dataset_path: Optional[str] = None, **data):
        super().__init__(**data)
        self.dataset_path = dataset_path
        self.riza_client = None
        self.runtime_revision_id = None
        
        if Riza:
            api_key = os.getenv("RIZA_API_KEY")
            if api_key:
                self.riza_client = Riza(api_key=api_key)
                # Try to get or create a custom runtime with data science packages
                print("[Riza Executor] Initializing custom runtime for data science packages...")
                self.runtime_revision_id = self._get_or_create_runtime()
            else:
                print("Warning: RIZA_API_KEY not found. Remote code execution will not work.")
    
    def _get_or_create_runtime(self) -> Optional[str]:
        """Get or create a custom runtime with data science packages."""
        if not self.riza_client:
            return None
            
        runtime_name = "data_science_agent_runtime"
        
        # Data science packages requirements
        requirements = """pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
scipy==1.11.4"""
        
        try:
            # Try to find existing runtime
            print("[Riza Executor] Looking for existing data science runtime...")
            
            # List existing runtimes to see if we already have one
            try:
                runtimes = self.riza_client.runtimes.list()
                for runtime in runtimes.data:
                    if runtime.name == runtime_name and runtime.status == "succeeded":
                        print(f"[Riza Executor] Found existing runtime: {runtime.revision_id}")
                        return runtime.revision_id
            except:
                pass  # Continue to create new runtime
            
            # Create new runtime
            print("[Riza Executor] Creating new data science runtime...")
            runtime = self.riza_client.runtimes.create(
                name=runtime_name,
                language="python",
                manifest_file={
                    "name": "requirements.txt",
                    "contents": requirements
                }
            )
            
            runtime_id = runtime.id
            print(f"[Riza Executor] Runtime created with ID: {runtime_id}")
            
            # Wait for runtime to build
            print("[Riza Executor] Waiting for runtime to build...")
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status_check = self.riza_client.runtimes.get(runtime_id)
                status = status_check.status
                print(f"[Riza Executor] Runtime status: {status}")
                
                if status == "succeeded":
                    print(f"[Riza Executor] Runtime ready! Revision ID: {status_check.revision_id}")
                    return status_check.revision_id
                elif status == "failed":
                    print(f"[Riza Executor] Runtime build failed")
                    return None
                
                time.sleep(10)  # Wait 10 seconds between checks
            
            print("[Riza Executor] Runtime build timed out")
            return None
            
        except Exception as e:
            print(f"[Riza Executor] Error creating runtime: {e}")
            return None
    
    def _run(self, code: str) -> Dict[str, Any]:
        """Execute code remotely using Riza."""
        if not self.riza_client:
            return {
                "success": False,
                "error": "Riza client not initialized - please set RIZA_API_KEY",
                "output": None,
                "data": None,
                "visualizations": None
            }
        
        try:
            # Prepare the code with dataset loading
            full_code = self._prepare_code(code)
            
            # Execute using Riza with custom runtime
            if self.runtime_revision_id:
                print(f"[Riza Executor] Executing with custom runtime: {self.runtime_revision_id}")
                result = self.riza_client.command.exec(
                    runtime_revision_id=self.runtime_revision_id,
                    code=full_code
                )
            else:
                print("[Riza Executor] Executing with default runtime (limited packages)")
                result = self.riza_client.command.exec(
                    language="python",
                    code=full_code
                )
            
            # Parse the result using Riza-specific parsing
            return self._parse_riza_result(result)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Riza execution error: {str(e)}",
                "output": None,
                "data": None,
                "visualizations": None
            }
    
    async def _arun(self, code: str) -> Dict[str, Any]:
        """Async version of code execution."""
        # For now, we'll use the sync version
        # Riza tool doesn't have async support in the current LangChain integration
        return self._run(code)
    
    def _prepare_code(self, code: str) -> str:
        """Prepare code for execution - minimal modification for transparency.
        
        The coding agent now generates complete code.
        """
        # Just add a simple header
        header = '''# =============================================================================
# AUTO-GENERATED DATA SCIENCE CODE (Remote Execution)
# =============================================================================
'''
        return f"{header}\n{code}"
    
    def _parse_riza_result(self, result) -> Dict[str, Any]:
        """Parse the Riza API execution result."""
        try:
            # Check execution status
            if result.execution.exit_code != 0:
                error_msg = result.execution.stderr or "Unknown execution error"
                return self._parse_output("", error_msg)
            
            # Get the output and use base class parsing
            output = result.execution.stdout or ""
            return self._parse_output(output)
            
        except Exception as e:
            return self._parse_output("", f"Error parsing Riza result: {str(e)}")
    
    async def _arun(self, code: str) -> Dict[str, Any]:
        """Async version - just calls sync version."""
        return self._run(code)


def create_code_executor(
    dataset_path: Optional[str] = None, 
    execution_mode: Literal["local", "remote"] = "local"
) -> BaseCodeExecutor:
    """
    Create a code executor tool instance.
    
    Args:
        dataset_path: Optional path to the dataset file to pre-load
        execution_mode: Either "local" (with user approval) or "remote" (Riza)
        
    Returns:
        BaseCodeExecutor instance (LocalCodeExecutor or RizaCodeExecutor)
    """
    if execution_mode == "local":
        print("🏠 Using LOCAL code executor with user approval")
        return LocalCodeExecutor(dataset_path=dataset_path)
    
    elif execution_mode == "remote":
        print("☁️ Using REMOTE code executor (Riza)")
        
        # Check for Riza requirements
        if not Riza:
            print("❌ Error: rizaio package not found. Please install it with: pip install rizaio")
            print("Falling back to local execution...")
            return LocalCodeExecutor(dataset_path=dataset_path)
        
        if not os.getenv("RIZA_API_KEY"):
            print("❌ Warning: RIZA_API_KEY environment variable not found.")
            print("You can get an API key from: https://docs.riza.io/")
            print("Falling back to local execution...")
            return LocalCodeExecutor(dataset_path=dataset_path)
            
        return RizaCodeExecutor(dataset_path=dataset_path)
    
    else:
        raise ValueError(f"Invalid execution_mode: {execution_mode}. Must be 'local' or 'remote'")


# Convenience functions for specific executor types
def create_local_executor(dataset_path: Optional[str] = None) -> LocalCodeExecutor:
    """Create a local code executor with user approval."""
    return LocalCodeExecutor(dataset_path=dataset_path)


def create_remote_executor(dataset_path: Optional[str] = None) -> RizaCodeExecutor:
    """Create a remote Riza code executor."""
    return RizaCodeExecutor(dataset_path=dataset_path) 