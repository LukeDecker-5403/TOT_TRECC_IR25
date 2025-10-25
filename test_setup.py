#!/usr/bin/env python3
"""Quick test to verify everything works"""

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    try:
        from tot_retrieval.config import Config
        print("  ✅ config.py")
        
        from tot_retrieval.data_loader import DataLoader
        print("  ✅ data_loader.py")
        
        from tot_retrieval.query_decomposer_free import RuleBasedQueryDecomposer
        print("  ✅ query_decomposer_free.py")
        
        from tot_retrieval.ensemble_retriever_pyserini import PyseriniEnsembleRetriever
        print("  ✅ ensemble_retriever_pyserini.py")
        
        from tot_retrieval.evaluation import TOTEvaluator
        print("  ✅ evaluation.py")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_java():
    """Test Java is available"""
    import subprocess
    print("\nTesting Java...")
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True)
        version = result.stderr.split('\n')[0]
        print(f"  ✅ {version}")
        return True
    except:
        print("  ❌ Java not found")
        return False

def test_config():
    """Test config validation"""
    print("\nTesting configuration...")
    try:
        from tot_retrieval.config import Config
        Config.validate()
        print("  ✅ Configuration valid")
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TOT Retrieval System - Setup Test")
    print("=" * 60)
    print()
    
    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_java()
    all_passed &= test_config()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ SETUP COMPLETE - Ready to use!")
        print("\nNext step: Run demo")
        print("  python3 -m tot_retrieval.main_lucene --mode demo")
    else:
        print("❌ Setup incomplete - fix errors above")
    print("=" * 60)
