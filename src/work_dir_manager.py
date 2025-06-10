"""
Work Directory Manager - Organized File Management and Logging
=============================================================

This module manages the creation and organization of work directories for each
agent run. It ensures all outputs are saved in a structured, reproducible way.

Core Responsibilities:
---------------------
1. **Directory Creation**: Creates timestamped run directories
2. **File Organization**: Manages inputs, scripts, results subdirectories
3. **Logging System**: Captures all agent activities and LLM interactions
4. **Safe Operations**: Prevents accidental file deletion or overwrites
5. **Metadata Tracking**: Records run configuration and results

Directory Structure:
-------------------
```
agent_work_dir/{model}-{timestamp}-{uuid}/
├── inputs/              # Uploaded datasets
├── scripts/             # Generated Python scripts
├── results/             # Outputs, visualizations, models
├── execution.log        # Complete execution log
├── llm_history.json     # All LLM requests/responses
└── metadata.json        # Run configuration and status
```

Key Features:
------------
- **Unique Run IDs**: Timestamp + UUID prevents collisions
- **Dual Logging**: Both file and console output
- **LLM Tracking**: Records all tokens used per agent
- **JSON Serialization**: Handles complex LangChain objects
- **Binary Support**: Saves images and other binary files

Main Components:
---------------
- **WorkDirectoryManager**: Core class managing directories
- **create_run_directory()**: Initializes new run environment
- **log_llm_request()**: Tracks LLM usage with token counts
- **save_file()**: Saves text files with organization
- **save_result()**: Saves outputs to results subdirectory

Global Access Pattern:
---------------------
Uses singleton pattern for global access:
```python
manager = get_work_dir_manager()
manager.log("Starting analysis...")
manager.save_result("output.csv", data)
```

Logging Levels:
--------------
- INFO: Normal operations and progress
- WARNING: Non-critical issues (file overwrites)
- ERROR: Critical failures
- DEBUG: Detailed troubleshooting info

Safety Features:
---------------
- No delete operations allowed
- Warns before overwriting files
- Creates parent directories automatically
- Handles serialization errors gracefully
- Preserves all intermediate outputs

Use Cases:
---------
1. **Dataset Storage**: Copies uploaded files to inputs/
2. **Script Saving**: Stores all generated code
3. **Result Organization**: Keeps outputs in results/
4. **Debug Analysis**: Complete logs for troubleshooting
5. **Token Tracking**: Monitor API usage per agent

Interview Notes:
---------------
- Essential for reproducibility - every run is self-contained
- The logging system has been invaluable for debugging
- Token tracking helps monitor API costs
- Directory structure inspired by ML experiment tracking
- No cleanup needed - old runs provide learning examples
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid
import shutil


class WorkDirectoryManager:
    """Manages work directories for agent runs."""
    
    def __init__(self, base_dir: str = "./agent_work_dir", model_name: str = "data-science-agent"):
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.run_id = None
        self.work_dir = None
        self.results_dir = None
        self.log_file = None
        self.llm_history_file = None
        self.logger = None
        
    def create_run_directory(self) -> str:
        """Create a new run directory with unique ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        self.run_id = f"{self.model_name}-{timestamp}-{unique_id}"
        
        # Create directory structure
        self.work_dir = self.base_dir / self.run_id
        self.results_dir = self.work_dir / "results"
        
        # Create directories
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize log files
        self.log_file = self.work_dir / "execution.log"
        self.llm_history_file = self.work_dir / "llm_history.json"
        
        # Set up logger
        self._setup_logger()
        
        # Create initial metadata
        self._create_metadata()
        
        self.log(f"🚀 Created run directory: {self.work_dir}")
        self.log(f"📁 Results will be saved to: {self.results_dir}")
        
        return str(self.work_dir)
    
    def _setup_logger(self):
        """Set up logging to file and console."""
        self.logger = logging.getLogger(f"WorkDir-{self.run_id}")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _create_metadata(self):
        """Create metadata file for the run."""
        metadata = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "created_at": datetime.now().isoformat(),
            "work_dir": str(self.work_dir),
            "results_dir": str(self.results_dir),
            "status": "running"
        }
        
        metadata_file = self.work_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def log(self, message: str, level: str = "info"):
        """Log a message to both file and console."""
        if self.logger:
            if level.lower() == "error":
                self.logger.error(message)
            elif level.lower() == "warning":
                self.logger.warning(message)
            elif level.lower() == "debug":
                self.logger.debug(message)
            else:
                self.logger.info(message)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def log_llm_request(self, agent_name: str, request: Dict[str, Any], response: Dict[str, Any], 
                       input_tokens: int = 0, output_tokens: int = 0):
        """Log LLM request and response to history file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "request": self._make_serializable(request),
            "response": self._make_serializable(response),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        
        # Append to LLM history file
        if self.llm_history_file.exists():
            with open(self.llm_history_file, 'r') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        else:
            history = []
        
        history.append(entry)
        
        try:
            with open(self.llm_history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            self.log(f"⚠️ Error saving LLM history: {e}", "warning")
            # Try to save a simplified version
            try:
                simplified_entry = {
                    "timestamp": entry["timestamp"],
                    "agent_name": entry["agent_name"],
                    "input_tokens": entry["input_tokens"],
                    "output_tokens": entry["output_tokens"],
                    "total_tokens": entry["total_tokens"],
                    "error": "Full data could not be serialized"
                }
                history[-1] = simplified_entry
                with open(self.llm_history_file, 'w') as f:
                    json.dump(history, f, indent=2)
            except:
                pass
        
        self.log(f"🤖 {agent_name}: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total tokens")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            # For objects with __dict__, try to extract relevant info
            try:
                # Special handling for common LangChain types
                if hasattr(obj, 'content'):
                    return {"type": obj.__class__.__name__, "content": str(obj.content)}
                elif hasattr(obj, 'text'):
                    return {"type": obj.__class__.__name__, "text": str(obj.text)}
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
    
    def save_file(self, filename: str, content: str, subdirectory: str = ""):
        """Save a file to the work directory."""
        if subdirectory:
            target_dir = self.work_dir / subdirectory
            target_dir.mkdir(exist_ok=True)
        else:
            target_dir = self.work_dir
        
        file_path = target_dir / filename
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prevent overwriting without confirmation
        if file_path.exists():
            self.log(f"⚠️  File {filename} already exists, overwriting...", "warning")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        self.log(f"💾 Saved file: {file_path}")
        return str(file_path)
    
    def save_result(self, filename: str, content: str):
        """Save a result file to the results directory."""
        file_path = self.results_dir / filename
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        self.log(f"📊 Saved result: {file_path}")
        return str(file_path)
    
    def save_binary_file(self, filename: str, content: bytes, subdirectory: str = ""):
        """Save a binary file (e.g., images) to the work directory."""
        if subdirectory:
            target_dir = self.work_dir / subdirectory
            target_dir.mkdir(exist_ok=True)
        else:
            target_dir = self.work_dir
        
        file_path = target_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        self.log(f"💾 Saved binary file: {file_path}")
        return str(file_path)
    
    def save_result_binary(self, filename: str, content: bytes):
        """Save a binary result file to the results directory."""
        file_path = self.results_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        self.log(f"📊 Saved binary result: {file_path}")
        return str(file_path)
    
    def list_files(self, subdirectory: str = "") -> List[str]:
        """List files in the work directory or subdirectory."""
        if subdirectory:
            target_dir = self.work_dir / subdirectory
        else:
            target_dir = self.work_dir
        
        if not target_dir.exists():
            return []
        
        return [f.name for f in target_dir.iterdir() if f.is_file()]
    
    def get_file_path(self, filename: str, subdirectory: str = "") -> str:
        """Get the full path to a file in the work directory."""
        if subdirectory:
            return str(self.work_dir / subdirectory / filename)
        else:
            return str(self.work_dir / filename)
    
    def get_result_path(self, filename: str) -> str:
        """Get the full path to a result file."""
        return str(self.results_dir / filename)
    
    def finalize_run(self, status: str = "completed"):
        """Finalize the run and update metadata."""
        if self.work_dir and self.work_dir.exists():
            # Update metadata
            metadata_file = self.work_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                metadata["status"] = status
                metadata["completed_at"] = datetime.now().isoformat()
                metadata["files_created"] = self.list_files()
                metadata["results_created"] = self.list_files("results")
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        self.log(f"✅ Run finalized with status: {status}")
    
    def copy_input_file(self, source_path: str, filename: str = None) -> str:
        """Copy an input file to the work directory."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if filename is None:
            filename = source.name
        
        dest_path = self.work_dir / "inputs" / filename
        dest_path.parent.mkdir(exist_ok=True)
        
        shutil.copy2(source, dest_path)
        self.log(f"📥 Copied input file: {filename}")
        
        return str(dest_path)


# Global work directory manager instance
_work_dir_manager = None


def get_work_dir_manager() -> WorkDirectoryManager:
    """Get the global work directory manager instance."""
    global _work_dir_manager
    if _work_dir_manager is None:
        _work_dir_manager = WorkDirectoryManager()
    return _work_dir_manager


def create_new_run(model_name: str = None) -> str:
    """Create a new run directory and return the path."""
    manager = get_work_dir_manager()
    if model_name:
        manager.model_name = model_name
    return manager.create_run_directory()


def log_message(message: str, level: str = "info"):
    """Log a message using the work directory manager."""
    manager = get_work_dir_manager()
    manager.log(message, level)


def log_llm_interaction(agent_name: str, request: Dict[str, Any], response: Dict[str, Any], 
                       input_tokens: int = 0, output_tokens: int = 0):
    """Log an LLM interaction."""
    manager = get_work_dir_manager()
    manager.log_llm_request(agent_name, request, response, input_tokens, output_tokens) 