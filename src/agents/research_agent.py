"""
Research Agent - Kaggle Competition Knowledge Retrieval
======================================================

This module implements the research agent that searches through 647 Kaggle competitions
to find similar problems and extract winning techniques. It's like having a Kaggle
grandmaster consultant who instantly recalls relevant competitions.

Core Functionality:
------------------
1. **Query Generation**: Creates multiple search queries from the problem description
2. **Vector Search**: Searches embeddings of Kaggle competition summaries
3. **Result Ranking**: Sorts results by relevance and deduplicates
4. **Insight Extraction**: Pulls out techniques, approaches, and key learnings
5. **Summary Creation**: Synthesizes findings for the coding agent

Search Strategy:
---------------
- Generates 3-5 diverse queries to cast a wide net
- Each query returns top 3 most similar competitions
- Results are merged, deduplicated, and re-ranked
- Top 5 competitions are analyzed for insights

Key Components:
--------------
- **generate_search_queries()**: Uses LLM to create smart search queries
- **KaggleVectorSearch**: Tool that searches pre-computed embeddings
- **extract_insights()**: Parses competition data for actionable techniques
- **rank_kaggle_solutions()**: Scores and sorts by relevance

Data Sources:
------------
The vector database contains:
- Competition titles and descriptions
- Problem types (classification, regression, etc.)
- Winning techniques and approaches
- Key insights from top solutions
- Evaluation metrics used

Technical Details:
-----------------
- Uses async/await for parallel searches
- Supports custom LLM models (defaults to GPT-4)
- Graceful fallbacks when structured output fails
- Debug mode shows detailed search progress

Interview Notes:
---------------
- This agent is READ-ONLY - it never modifies the vector database
- The quality of results depends on the vector embeddings quality
- Falls back to generic queries if LLM fails to generate good ones
- Results include similarity scores (0-1) for transparency
- The agent passes ALL findings to state, orchestrator decides what to use
"""

from typing import Dict, Any, List
import os
import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from state import DataScienceState
from prompts import RESEARCH_AGENT_PROMPT, QUERY_GENERATION_PROMPT
from tools.vector_search import create_vector_search_tool
from llm_utils import get_default_llm

# Set up logger
logger = logging.getLogger(__name__)

def debug_print(message: str):
    """Debug print that only shows in debug mode."""
    if os.getenv("DEBUG_MODE"):
        print(f"🐛 [RESEARCH] {message}")
    logger.debug(f"[RESEARCH] {message}")

class ResearchQueries(BaseModel):
    """Structured output for search query generation"""
    queries: List[str] = Field(
        description="List of search queries for the vector database"
    )
    reasoning: str = Field(
        description="Reasoning behind the query choices"
    )

class ResearchFindings(BaseModel):
    """Structured output for research findings"""
    relevant_competitions: List[Dict[str, str]] = Field(
        description="List of relevant Kaggle competitions found"
    )
    key_techniques: List[str] = Field(
        description="Important techniques and approaches discovered"
    )
    recommended_approach: str = Field(
        description="Recommended approach based on research"
    )

def create_research_node(callbacks=None, custom_model=None):
    """
    Creates the research agent node.
    Searches Kaggle database for relevant insights.
    
    Args:
        callbacks: Optional callbacks for streaming
        custom_model: Dict with custom model config:
                     {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"}
    
    Returns:
        Callable: Node function for LangGraph
    """
    async def research_agent(state: DataScienceState) -> Dict[str, Any]:
        """
        Research agent logic.
        
        Steps:
        1. Analyze problem and generate search queries
        2. Search vector database for similar competitions
        3. Extract and rank insights
        4. Return structured findings
        """
        print(f"[Research Agent] Searching for relevant Kaggle competitions and insights...")
        debug_print("Research agent started")
        debug_print(f"Input state keys: {list(state.keys())}")
        
        # Initialize LLM - use custom model if provided
        debug_print("Initializing LLM for research agent...")
        if custom_model:
            debug_print(f"Using custom model from parameters: {custom_model}")
            from llm_utils import create_custom_llm
            llm = create_custom_llm(
                provider=custom_model["provider"],
                model_id=custom_model["model"],
                temperature=0.0,
                streaming=True,
                callbacks=callbacks,
                agent_name="🔍 Research"
            )
        elif os.getenv("RESEARCH_PROVIDER") and os.getenv("RESEARCH_MODEL"):
            # Check for command-line environment variables
            provider = os.getenv("RESEARCH_PROVIDER")
            model = os.getenv("RESEARCH_MODEL")
            debug_print(f"Using custom model from environment: {provider}/{model}")
            from llm_utils import create_custom_llm
            llm = create_custom_llm(
                provider=provider,
                model_id=model,
                temperature=0.0,
                streaming=True,
                callbacks=callbacks,
                agent_name="🔍 Research"
            )
        else:
            debug_print("Using default LLM")
            llm = get_default_llm(temperature=0, streaming=True, callbacks=callbacks)
        debug_print("LLM initialized successfully")
        
        # Get search tool
        debug_print("Creating vector search tool...")
        search_tool = create_vector_search_tool()
        debug_print("Vector search tool created")
        
        # Generate search queries
        debug_print("Generating search queries...")
        problem_desc = state.get("problem_description", "")
        dataset_info = state.get("dataset_info", {})
        debug_print(f"Problem description: {problem_desc[:100]}...")
        debug_print(f"Dataset info: {dataset_info}")
        
        queries = await generate_search_queries(llm, problem_desc, dataset_info)
        debug_print(f"Generated {len(queries)} search queries: {queries}")
        
        # Execute searches
        debug_print("Starting vector database searches...")
        all_results = []
        print(f"[Research Agent] Generated {len(queries)} search queries")
        
        for i, query in enumerate(queries, 1):
            try:
                print(f"[Research Agent] Executing search {i}/{len(queries)}: '{query[:50]}...'")
                debug_print(f"Executing search {i}: '{query}'")
                
                results = await search_tool.ainvoke({"query": query, "k": 3})
                debug_print(f"Search {i} returned {len(results)} results")
                
                all_results.extend(results)
            except Exception as e:
                print(f"[Research Agent] Search error for '{query}': {e}")
                debug_print(f"Search error for query '{query}': {e}")
        
        debug_print(f"Total search results collected: {len(all_results)}")
        
        # Rank and deduplicate results
        debug_print("Ranking and deduplicating results...")
        ranked_results = rank_kaggle_solutions(
            all_results, 
            state.get("problem_description", "")
        )
        debug_print(f"Ranked results: {len(ranked_results)} unique competitions")
        
        # Extract insights
        debug_print("Extracting insights from top results...")
        insights = extract_insights(ranked_results[:5])  # Top 5 results
        debug_print(f"Extracted {len(insights)} insights")
        
        # Summarize findings
        debug_print("Summarizing research findings...")
        summary = await summarize_research(llm, insights, state)
        debug_print(f"Research summary generated: {summary[:100]}...")
        
        # Update state
        return {
            "kaggle_insights": insights,
            "search_queries": queries,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Found {len(insights)} relevant Kaggle competitions. {summary}")
            ]
        }
    
    return research_agent

async def generate_search_queries(llm, problem: str, dataset_info: Dict) -> List[str]:
    """
    Generate diverse search queries for the vector database with fallback parsing.
    """
    debug_print("Starting query generation...")
    debug_print(f"Problem: {problem[:100]}...")
    debug_print(f"Dataset info: {dataset_info}")
    
    # Format the prompt
    debug_print("Formatting query generation prompt...")
    prompt = QUERY_GENERATION_PROMPT.format(
        problem=problem,
        dataset_info=dataset_info,
        num_queries=3
    )
    debug_print(f"Prompt created, length: {len(prompt)} characters")
    
    # Generate queries with fallback
    debug_print("Generating queries with LLM...")
    messages = [
        SystemMessage(content="You are a search query generator for finding relevant Kaggle competitions."),
        HumanMessage(content=f"{prompt}\n\nRespond with your queries in this format:\nQUERIES:\n1. [first query]\n2. [second query]\n3. [third query]\nREASONING: [your reasoning]")
    ]
    debug_print(f"Calling LLM with {len(messages)} messages...")
    
    try:
        # Try structured output first
        debug_print("Attempting structured output...")
        structured_llm = llm.with_structured_output(ResearchQueries)
        result = await structured_llm.ainvoke(messages)
        debug_print("Structured output successful")
        debug_print(f"Generated queries: {result.queries}")
        return result.queries
        
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
                debug_print("=== FULL RESEARCH LLM RESPONSE FOR DEBUGGING ===")
                debug_print(text)
                debug_print("=== END FULL RESEARCH RESPONSE ===")
            
            # Parse the response
            queries = parse_search_queries_response(text)
            debug_print("Text parsing successful")
            debug_print(f"Parsed queries: {queries}")
            return queries
            
        except Exception as parse_error:
            debug_print(f"Text parsing failed: {str(parse_error)}")
            # Ultimate fallback with generic queries
            return get_fallback_queries(problem)


def parse_search_queries_response(text: str) -> List[str]:
    """
    Parse text response into search queries.
    
    Args:
        text: Raw text response from LLM
        
    Returns:
        List of search queries
    """
    queries = []
    
    # Look for numbered list patterns
    query_patterns = [
        r'(?:QUERIES:)?\s*\n?\s*(\d+\.?\s*[^\n]+)',  # "1. query text"
        r'(?:Query \d+:)\s*([^\n]+)',                 # "Query 1: text"
        r'(?:-\s*)([^\n]+)',                          # "- query text"
    ]
    
    for pattern in query_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches and len(matches) >= 2:  # Need at least 2 queries
            queries = [match.strip() for match in matches]
            break
    
    # If no patterns worked, try to extract lines that look like queries
    if not queries:
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines, reasoning sections, etc.
            if (line and 
                not line.lower().startswith(('reasoning:', 'rationale:', 'explanation:')) and
                len(line) > 10 and 
                any(keyword in line.lower() for keyword in ['predict', 'classify', 'analysis', 'model', 'data', 'competition'])):
                # Clean up the line
                cleaned = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                cleaned = re.sub(r'^-\s*', '', cleaned)    # Remove dashes
                if len(cleaned) > 10:
                    queries.append(cleaned)
    
    # Ensure we have at least some queries
    if not queries:
        queries = ["machine learning classification prediction", "data science competition analysis", "feature engineering techniques"]
    
    # Limit to reasonable number
    return queries[:5]


def get_fallback_queries(problem: str) -> List[str]:
    """
    Generate fallback queries based on simple problem analysis.
    
    Args:
        problem: Problem description
        
    Returns:
        List of fallback search queries
    """
    debug_print("Using fallback query generation")
    
    queries = []
    problem_lower = problem.lower()
    
    # Detect problem type and add relevant queries
    if any(word in problem_lower for word in ['predict', 'classification', 'classify']):
        queries.append("classification prediction machine learning")
    elif any(word in problem_lower for word in ['regression', 'forecast', 'continuous']):
        queries.append("regression prediction continuous values")
    elif any(word in problem_lower for word in ['cluster', 'segment', 'group']):
        queries.append("clustering segmentation unsupervised learning")
    else:
        queries.append("data science machine learning analysis")
    
    # Add domain-specific queries
    if any(word in problem_lower for word in ['image', 'vision', 'photo']):
        queries.append("computer vision image classification")
    elif any(word in problem_lower for word in ['text', 'nlp', 'language']):
        queries.append("natural language processing text analysis")
    elif any(word in problem_lower for word in ['time', 'series', 'temporal']):
        queries.append("time series forecasting analysis")
    
    # Add general data science query
    queries.append("feature engineering data preprocessing")
    
    return queries[:3]

def rank_kaggle_solutions(results: List[Dict], problem: str) -> List[Dict]:
    """
    Rank and deduplicate Kaggle solutions by relevance.
    Simple scoring based on similarity and problem match.
    """
    # Deduplicate by title
    seen_titles = set()
    unique_results = []
    
    for result in results:
        title = result.get("title", "")
        if title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(result)
    
    # Sort by similarity score
    unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return unique_results

def extract_insights(solutions: List[Dict]) -> List[Dict[str, str]]:
    """
    Extract structured insights from Kaggle solutions.
    """
    insights = []
    
    for solution in solutions:
        insight = {
            "title": solution.get("title", "Unknown Competition"),
            "score": solution.get("score", 0.0),
            "summary": solution.get("summary", ""),
            "techniques": [],
            "key_points": []
        }
        
        # Add full problem content if available
        if "problem_summary" in solution:
            insight["problem_summary"] = solution["problem_summary"]
        if "problem_id" in solution:
            insight["problem_id"] = solution["problem_id"]
        if "objective" in solution:
            insight["objective"] = solution["objective"]
        
        # Extract techniques from key_insights
        if "key_insights" in solution:
            for item in solution["key_insights"]:
                if any(keyword in item.lower() for keyword in ["approach", "technique", "method", "model"]):
                    insight["techniques"].append(item)
                else:
                    insight["key_points"].append(item)
        
        # Also extract techniques from techniques field
        if "techniques" in solution:
            insight["techniques"].extend(solution["techniques"])
        
        # Deduplicate techniques
        insight["techniques"] = list(set(insight["techniques"]))
        
        insights.append(insight)
    
    return insights

async def summarize_research(llm, insights: List[Dict], state: DataScienceState) -> str:
    """
    Create a summary of research findings.
    """
    if not insights:
        return "No relevant Kaggle competitions found. Will proceed with general best practices."
    
    # Format insights for summary
    insights_text = []
    for i, insight in enumerate(insights[:3], 1):
        insights_text.append(
            f"{i}. {insight['title']} (score: {insight['score']:.2f})\n"
            f"   Techniques: {', '.join(insight['techniques'][:3])}"
        )
    
    summary_prompt = f"""
    Based on these Kaggle competition insights:
    {chr(10).join(insights_text)}
    
    Problem: {state.get('problem_description', '')}
    
    Provide a brief summary of the most relevant approaches to try.
    """
    
    messages = [
        SystemMessage(content="Summarize the research findings concisely."),
        HumanMessage(content=summary_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    return response.content 


# Standalone function for testing
async def run_research_agent(problem_description: str, dataset_info: str) -> List[Dict[str, Any]]:
    """
    Run the research agent standalone for testing.
    
    Args:
        problem_description: Problem to search for
        dataset_info: Dataset description
        
    Returns:
        List of Kaggle insights
    """
    # Create minimal state
    state = {
        "problem_description": problem_description,
        "dataset_info": {"description": dataset_info} if isinstance(dataset_info, str) else dataset_info
    }
    
    # Create LLM
    llm = get_default_llm(temperature=0)
    
    # Generate search queries
    queries = await generate_search_queries(
        llm,
        problem_description,
        state["dataset_info"]
    )
    
    # Create search tool
    from tools.vector_search import KaggleVectorSearch
    searcher = KaggleVectorSearch()
    
    # Execute searches
    all_results = []
    for query in queries[:2]:  # Limit queries for testing
        try:
            results = searcher.search(query, k=3)
            all_results.extend(results)
        except Exception as e:
            print(f"Search error: {e}")
    
    # Format as insights
    insights = []
    for r in all_results[:5]:  # Top 5 results
        insights.append({
            "competition_name": r.get("competition_name", "Unknown"),
            "problem_type": r.get("problem_type", ""),
            "ml_techniques": r.get("ml_techniques", []),
            "key_insights": r.get("key_insights", ""),
            "similarity_score": r.get("similarity_score", 0.0)
        })
    
    return insights 