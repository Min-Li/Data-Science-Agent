"""
Vector Search Tool - Kaggle Competition Knowledge Base
=====================================================

This module implements the vector search functionality for finding similar Kaggle
competitions based on semantic similarity. It's the bridge between the research
agent's queries and the pre-computed embeddings database.

Core Functionality:
------------------
1. **Embedding Loading**: Loads pre-computed embeddings from .npy files
2. **Query Embedding**: Converts search queries to embeddings using OpenAI
3. **Similarity Search**: Finds most similar competitions using cosine similarity
4. **Result Enrichment**: Adds full problem summaries and techniques
5. **LangChain Integration**: Provides tool interface for agents

Data Sources:
------------
The tool expects pre-computed data in this structure:
```
data/vector_db/{embedding_model}/
├── embeddings.npy      # NumPy array of competition embeddings
├── metadata.csv        # Competition details and metadata
└── problem_summaries.json  # Full competition descriptions
```

Supported Embedding Models:
--------------------------
- openai-text-embedding-3-small (default, most efficient)
- openai-text-embedding-3-large (higher quality)
- openai-text-embedding-ada-002 (legacy)
- google-text-embedding-004
- google-gemini-embedding-exp-03-07

Key Components:
--------------
- **KaggleVectorSearch**: Main class handling search operations
- **search()**: Synchronous search method
- **asearch()**: Async wrapper for LangGraph compatibility
- **create_vector_search_tool()**: Creates LangChain tool interface

Search Process:
--------------
1. User query → Embed using same model as database
2. Compare against all competition embeddings
3. Sort by cosine similarity score
4. Return top-k results with metadata
5. Enrich with full problem summaries if available

Result Format:
-------------
Each result includes:
- Competition name and ID
- Problem type (classification, regression, etc.)
- ML techniques used by winners
- Key insights and approaches
- Similarity score (0-1)
- Full problem description (if available)

Performance Notes:
-----------------
- Embeddings are loaded into memory (fast search, ~500MB RAM)
- First query initializes OpenAI client (slight delay)
- Subsequent queries are very fast (<100ms)
- Falls back gracefully if vector database not found

Interview Notes:
---------------
- This is a READ-ONLY tool - never modifies the database
- Quality depends on pre-computed embeddings quality
- Requires OPENAI_API_KEY for query embedding
- Can work without vector DB (returns empty results)
- Debug mode shows detailed search progress
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import os
import logging
import json
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore

# Set up logger
logger = logging.getLogger(__name__)

def debug_print(message: str):
    """Debug print that only shows in debug mode."""
    if os.getenv("DEBUG_MODE"):
        print(f"🐛 [VECTOR_SEARCH] {message}")
    logger.debug(f"[VECTOR_SEARCH] {message}")


class KaggleVectorSearch:
    """
    Vector search implementation for Kaggle competition database.
    Simplified implementation that loads pre-computed embeddings.
    """
    def __init__(self, embeddings_path: str = "data/embeddings.npy", 
                 metadata_path: str = "data/metadata.csv",
                 embedding_model: str = "openai-text-embedding-3-small"):
        """Initialize the vector search tool."""
        debug_print("Initializing KaggleVectorSearch...")
        debug_print(f"Embeddings path: {embeddings_path}")
        debug_print(f"Metadata path: {metadata_path}")
        debug_print(f"Embedding model: {embedding_model}")
        
        self._embedding_model = None  # Lazy initialization
        self.embedding_model_name = embedding_model
        self.embeddings = None
        self.metadata = None
        self.problem_summaries = None  # Store loaded problem summaries
        
        # Load credentials first
        debug_print("Loading credentials...")
        try:
            from llm_utils import load_credentials
            credentials = load_credentials()
            debug_print(f"Loaded {len(credentials)} credentials")
        except Exception as e:
            debug_print(f"Failed to load credentials: {e}")
        
        # Load problem summaries if available
        debug_print("Loading problem summaries...")
        try:
            summaries_path = Path(__file__).parent.parent.parent / "data" / "problem_summaries.json"
            if summaries_path.exists():
                with open(summaries_path, 'r') as f:
                    self.problem_summaries = json.load(f)
                debug_print(f"Loaded {len(self.problem_summaries)} problem summaries")
            else:
                debug_print(f"Problem summaries not found at {summaries_path}")
        except Exception as e:
            debug_print(f"Failed to load problem summaries: {e}")
        
        # Determine vector_db subdirectory based on model
        vector_db_mapping = {
            "openai-text-embedding-3-small": "openai-text-embedding-3-small",
            "openai-text-embedding-3-large": "openai-text-embedding-3-large", 
            "openai-text-embedding-ada-002": "openai-text-embedding-ada-002",
            "google-text-embedding-004": "google-text-embedding-004",
            "google-gemini-embedding-exp-03-07": "google-gemini-embedding-exp-03-07"
        }
        
        # Get the subdirectory for this embedding model
        model_subdir = vector_db_mapping.get(embedding_model, "openai-text-embedding-3-small")
        
        # Define the exact location where vector databases should be
        agent_dir = Path(__file__).parent.parent.parent  # open_data_science_agent directory
        vector_db_base = agent_dir / "data" / "vector_db"
        
        # Look for embeddings in the correct location
        target_dir = vector_db_base / model_subdir
        emb_path = target_dir / "embeddings.npy"
        meta_path = target_dir / "metadata.csv"
        
        debug_print(f"Looking for vector database in: {target_dir}")
        
        if emb_path.exists() and meta_path.exists():
            try:
                self.embeddings = np.load(emb_path)
                self.metadata = pd.read_csv(meta_path)
                print(f"✅ Loaded {len(self.metadata)} Kaggle competitions from {target_dir.name}")
                debug_print(f"Successfully loaded {self.embeddings.shape} embeddings and {len(self.metadata)} metadata entries")
                debug_print(f"Embeddings shape: {self.embeddings.shape}")
                debug_print(f"Metadata columns: {list(self.metadata.columns)}")
            except Exception as e:
                debug_print(f"Failed to load from {target_dir}: {e}")
                self.embeddings = None
                self.metadata = None
        else:
            # Don't fail completely - just work without Kaggle data
            debug_print(f"Vector database not found in {target_dir}")
            
            # Check alternative locations
            project_root = Path(__file__).parent.parent.parent.parent  # Go up to DataScienceAgent
            alt_locations = [
                project_root / "openai_kaggle_db",
                project_root / f"openai_kaggle_db_{model_subdir}",
                project_root / f"{model_subdir}_kaggle_db"
            ]
            
            found_alternative = False
            for alt_path in alt_locations:
                alt_emb = alt_path / "embeddings.npy"
                alt_meta = alt_path / "metadata.csv"
                if alt_emb.exists() and alt_meta.exists():
                    try:
                        self.embeddings = np.load(alt_emb)
                        self.metadata = pd.read_csv(alt_meta)
                        print(f"✅ Loaded {len(self.metadata)} Kaggle competitions from {alt_path.name}")
                        debug_print(f"Found in alternative location: {alt_path}")
                        found_alternative = True
                        break
                    except Exception as e:
                        debug_print(f"Failed to load from {alt_path}: {e}")
            
            if not found_alternative:
                print(f"\n⚠️  No Kaggle vector database found for model: {self.embedding_model_name}")
                print(f"   Searched locations:")
                print(f"   - {target_dir}/")
                for alt_path in alt_locations:
                    print(f"   - {alt_path}/")
                
                # List available models if any exist
                if vector_db_base.exists():
                    available = []
                    for subdir in vector_db_base.iterdir():
                        if subdir.is_dir() and (subdir / "embeddings.npy").exists():
                            available.append(subdir.name)
                    
                    if available:
                        print(f"\n   Available models in {vector_db_base.name}:")
                        for model in available:
                            print(f"   - {model}")
                
                print("\n   The agent will work but won't have Kaggle competition insights.")
                self.embeddings = None
                self.metadata = None
    
    @property
    def embedding_model(self):
        """Lazy initialization of embedding model."""
        if self._embedding_model is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is required for embedding queries")
            # Remove "openai-" prefix from model name if present
            model_name = self.embedding_model_name
            if model_name.startswith("openai-"):
                model_name = model_name.replace("openai-", "")
            self._embedding_model = OpenAIEmbeddings(model=model_name)
        return self._embedding_model
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar Kaggle competitions (synchronous version).
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of competition insights
        """
        debug_print(f"Starting search for query: '{query}' (k={k})")
        
        # If no embeddings loaded, return empty results
        if self.embeddings is None or self.metadata is None:
            debug_print("No embeddings or metadata loaded, returning empty results")
            return []
        
        try:
            # Embed the query
            debug_print("Getting embedding model...")
            embedding_model = self.embedding_model
            debug_print("Embedding model obtained, generating query embedding...")
            
            query_embedding = embedding_model.embed_query(query)
            debug_print(f"Query embedded, shape: {np.array(query_embedding).shape}")
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            # Calculate similarities
            debug_print(f"Calculating similarities with {self.embeddings.shape[0]} competitions...")
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            debug_print(f"Similarities calculated, max: {similarities.max():.4f}, min: {similarities.min():.4f}")
            
            # Get top k indices
            debug_print(f"Finding top {k} results...")
            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_k_scores = similarities[top_k_indices]
            debug_print(f"Top scores: {top_k_scores}")
            
            # Build results
            debug_print("Building results...")
            results = []
            for idx, score in zip(top_k_indices, top_k_scores):
                row = self.metadata.iloc[idx]
                
                result = {
                    "competition_name": row.get('competition_name', 'Unknown'),
                    "problem_type": row.get('problem_type', 'Unknown'),
                    "dataset_description": row.get('dataset_description', ''),
                    "evaluation_metric": row.get('evaluation_metric', ''),
                    "winner_summary": row.get('winner_summary', ''),
                    "key_insights": row.get('key_insights', ''),
                    "ml_techniques": self._extract_techniques(row),
                    "similarity_score": float(score)
                }
                
                # Try to get problem ID from dataset_index
                dataset_index = row.get('dataset_index', None)
                if dataset_index is not None:
                    problem_id = str(dataset_index)
                    
                    # Add full problem summary if available
                    if self.problem_summaries and problem_id in self.problem_summaries:
                        summary_data = self.problem_summaries[problem_id]
                        result['problem_summary'] = summary_data.get('full_content', '')
                        result['problem_id'] = problem_id
                        
                        # Add parsed sections if available
                        if 'objective' in summary_data and summary_data['objective']:
                            result['objective'] = summary_data['objective']
                        if 'evaluation_metric' in summary_data and summary_data['evaluation_metric']:
                            result['evaluation_metric'] = summary_data['evaluation_metric']
                        
                        debug_print(f"Added full problem summary for problem {problem_id}")
                    else:
                        debug_print(f"No problem summary found for dataset_index {dataset_index}")
                
                results.append(result)
            
            debug_print(f"Search completed, returning {len(results)} results")
            return results
            
        except Exception as e:
            debug_print(f"Error during search: {e}")
            print(f"Error during search: {e}")
            import traceback
            debug_print(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def _extract_techniques(self, row: pd.Series) -> List[str]:
        """Extract ML techniques from competition data."""
        techniques = []
        
        # Try to extract from various fields
        for field in ['ml_techniques', 'techniques', 'winner_solution', 'winner_summary']:
            if field in row and pd.notna(row[field]):
                text = str(row[field])
                # Simple extraction of common ML terms
                ml_terms = ['XGBoost', 'Random Forest', 'Neural Network', 'LSTM', 'CNN',
                           'Gradient Boosting', 'SVM', 'Logistic Regression', 'Linear Regression',
                           'K-Means', 'PCA', 'Feature Engineering', 'Ensemble']
                
                for term in ml_terms:
                    if term.lower() in text.lower():
                        techniques.append(term)
        
        return list(set(techniques))[:5]  # Return unique techniques
    
    async def asearch(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Async wrapper for search method."""
        return self.search(query, k)

def create_vector_search_tool(embeddings_path: str = "data/embeddings.npy",
                             metadata_path: str = "data/metadata.csv",
                             embedding_model: str = "openai-text-embedding-3-small"):
    """
    Create a LangChain tool for vector search.
    
    Args:
        embeddings_path: Path to embeddings file
        metadata_path: Path to metadata file
        embedding_model: Name of the embedding model to use
        
    Returns:
        LangChain tool for searching Kaggle competitions
    """
    # Initialize the search instance
    searcher = KaggleVectorSearch(embeddings_path, metadata_path, embedding_model)
    
    @tool
    async def search_kaggle_solutions(
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar Kaggle competition solutions.
        
        Args:
            query: Problem description to search for
            k: Number of results to return (default: 5)
            filter_dict: Optional filters (not implemented yet)
            
        Returns:
            List of relevant Kaggle competition insights
        """
        debug_print(f"LangChain tool called with query: '{query}', k={k}")
        
        try:
            debug_print("Calling searcher.asearch...")
            results = await searcher.asearch(query, k)
            debug_print(f"Searcher returned {len(results)} results")
            
            # Format results for agent consumption
            debug_print("Formatting results for agent consumption...")
            formatted_results = []
            for i, r in enumerate(results):
                formatted_result = {
                    "title": r.get("competition_name", "Unknown"),
                    "score": r.get("similarity_score", 0.0),
                    "summary": f"{r.get('problem_type', '')}: {r.get('dataset_description', '')}",
                    "key_insights": r.get("ml_techniques", []),
                    "techniques": r.get("ml_techniques", [])
                }
                
                # Add full problem summary and additional fields if available
                if 'problem_summary' in r:
                    formatted_result['problem_summary'] = r['problem_summary']
                if 'problem_id' in r:
                    formatted_result['problem_id'] = r['problem_id']
                if 'objective' in r and r['objective']:
                    formatted_result['objective'] = r['objective']
                    
                formatted_results.append(formatted_result)
                debug_print(f"Formatted result {i+1}: {formatted_result['title']} (score: {formatted_result['score']:.4f})")
            
            debug_print(f"Returning {len(formatted_results)} formatted results")
            return formatted_results
            
        except Exception as e:
            debug_print(f"Error in vector search tool: {e}")
            print(f"Error in vector search: {e}")
            import traceback
            debug_print(f"Full traceback: {traceback.format_exc()}")
            return [{
                "title": "Error",
                "score": 0.0,
                "summary": f"Search failed: {str(e)}",
                "key_insights": [],
                "index": -1
            }]
    
    return search_kaggle_solutions

def extract_key_insights(content: str) -> List[str]:
    """
    Extract key insights from competition summary.
    Simple implementation - can be enhanced with NLP.
    """
    insights = []
    
    # Look for key sections
    sections_to_extract = [
        "Problem Type:",
        "Evaluation Metric:",
        "Key Features:",
        "Winning Approach:",
        "Important Techniques:"
    ]
    
    lines = content.split('\n')
    for line in lines:
        for section in sections_to_extract:
            if section in line:
                insights.append(line.strip())
                break
    
    # Also look for bullet points with techniques
    for line in lines:
        if line.strip().startswith(('- ', '* ', '• ')) and len(line.strip()) > 10:
            insights.append(line.strip())
            if len(insights) > 10:  # Limit insights
                break
    
    return insights[:8]  # Return top 8 insights

# Alternative implementation using LangChain's InMemoryVectorStore
def create_langchain_vector_store(db_path: str) -> InMemoryVectorStore:
    """
    Create a LangChain InMemoryVectorStore from the Kaggle database.
    Alternative approach using LangChain's built-in vector store.
    """
    # Load data
    embeddings_file = Path(db_path) / "embeddings.npy" 
    metadata_file = Path(db_path) / "metadata.csv"
    
    embeddings_array = np.load(embeddings_file)
    metadata_df = pd.read_csv(metadata_file)
    
    # Create documents
    documents = []
    for idx, row in metadata_df.iterrows():
        # Load summary content
        summary_path = row.get('summary_file_path', '')
        content = f"Competition: {row.get('competition_name', 'Unknown')}\n"
        
        if summary_path and os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    content += f.read()
            except:
                content += "Summary not available"
        
        doc = Document(
            page_content=content,
            metadata={
                "competition_name": row.get('competition_name', ''),
                "dataset_index": row.get('dataset_index', idx),
                "id": row.get('id', ''),
            }
        )
        documents.append(doc)
    
    # Create vector store
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embedding_model)
    
    # Add documents with pre-computed embeddings
    vector_store.add_documents(documents)
    
    return vector_store 