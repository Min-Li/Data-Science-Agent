#!/usr/bin/env python3
"""
Setup script to check and verify vector database location for the Data Science Agent.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def check_vector_db():
    """Check the vector database setup."""
    print("=" * 60)
    print("🔧 Vector Database Check for Data Science Agent")
    print("=" * 60)
    
    # Paths
    agent_dir = Path(__file__).parent
    data_dir = agent_dir / "data"
    vector_db_dir = data_dir / "vector_db"
    
    print(f"\n📍 Expected vector database location:")
    print(f"   {vector_db_dir}")
    
    if not vector_db_dir.exists():
        print(f"\n❌ Vector database directory not found!")
        print(f"   Please ensure the vector databases are in: {vector_db_dir}")
        print("\n   Expected structure:")
        print("   data/vector_db/")
        print("   ├── openai-text-embedding-3-small/")
        print("   │   ├── embeddings.npy")
        print("   │   └── metadata.csv")
        print("   ├── google-text-embedding-004/")
        print("   │   ├── embeddings.npy")
        print("   │   └── metadata.csv")
        print("   └── ... (other embedding models)")
        return False
    
    print("\n📊 Found vector databases:")
    
    # Check each subdirectory
    valid_dbs = []
    for subdir in sorted(vector_db_dir.iterdir()):
        if subdir.is_dir():
            emb_file = subdir / "embeddings.npy"
            meta_file = subdir / "metadata.csv"
            
            if emb_file.exists() and meta_file.exists():
                try:
                    # Load to check validity
                    emb = np.load(emb_file, mmap_mode='r')
                    meta = pd.read_csv(meta_file)
                    
                    # Get file sizes
                    emb_size = emb_file.stat().st_size / (1024 * 1024)  # MB
                    meta_size = meta_file.stat().st_size / 1024  # KB
                    
                    print(f"\n   ✅ {subdir.name}/")
                    print(f"      - embeddings.npy: {emb_size:.1f} MB")
                    print(f"      - Shape: {emb.shape}")
                    print(f"      - Dimensions: {emb.shape[1]}")
                    print(f"      - metadata.csv: {meta_size:.1f} KB")
                    print(f"      - Competitions: {len(meta)}")
                    
                    valid_dbs.append(subdir.name)
                    
                except Exception as e:
                    print(f"\n   ❌ {subdir.name}/ - Invalid database: {e}")
            else:
                missing = []
                if not emb_file.exists():
                    missing.append("embeddings.npy")
                if not meta_file.exists():
                    missing.append("metadata.csv")
                print(f"\n   ⚠️  {subdir.name}/ - Missing files: {', '.join(missing)}")
    
    if not valid_dbs:
        print("\n❌ No valid vector databases found!")
        return False
    
    print(f"\n✅ Found {len(valid_dbs)} valid vector database(s)")
    
    # Check credentials
    print("\n📋 Checking API credentials...")
    
    try:
        # Add src to path
        src_path = agent_dir / "src"
        sys.path.insert(0, str(src_path))
        
        from llm_utils import load_credentials
        credentials = load_credentials()
        
        print(f"   ✅ Loaded {len(credentials)} API keys")
        
        # Check for OpenAI key specifically (needed for embeddings)
        if "OPENAI_API_KEY" in credentials or os.getenv("OPENAI_API_KEY"):
            print("   ✅ OpenAI API key found (required for search)")
        else:
            print("   ⚠️  No OpenAI API key found - search functionality will be limited")
        
        # Check other keys
        for key in ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"]:
            if key in credentials or os.getenv(key):
                provider = key.replace("_API_KEY", "")
                print(f"   ✅ {provider} API key found")
                
    except Exception as e:
        print(f"   ❌ Failed to load credentials: {e}")
        print("   Make sure credentials.txt exists in the parent directory")
    
    # Recommendations
    print("\n📝 Recommendations:")
    
    if "openai-text-embedding-3-small" in valid_dbs:
        print("   ✅ Default embedding model (openai-text-embedding-3-small) is available")
    else:
        print("   ⚠️  Default embedding model not found")
        if valid_dbs:
            print(f"   You can use: {valid_dbs[0]}")
    
    print("\n✅ Vector database check completed!")
    
    # Test import
    print("\n🧪 Testing import...")
    try:
        from tools.vector_search import KaggleVectorSearch
        searcher = KaggleVectorSearch()
        if searcher.embeddings is not None:
            print("   ✅ Successfully loaded vector database in test import")
        else:
            print("   ⚠️  Vector database not loaded (but module imported successfully)")
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        return False
    
    return True

def main():
    """Main entry point."""
    print("This script checks the vector database setup for the Data Science Agent.\n")
    
    success = check_vector_db()
    
    if success:
        print("\n✅ Everything looks good! You can now run the agent.")
        print("\nTo test further, run:")
        print("   python test_vector_db_loading.py")
        print("\nTo run the agent:")
        print("   cd app && python -m streamlit run streamlit_app.py")
    else:
        print("\n❌ Setup issues detected. Please check the messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 