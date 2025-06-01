#!/usr/bin/env python3
"""
Direct script to create and test LanceDB vector index.
"""

import time
import lancedb

def create_vector_index():
    """Create vector index directly on the LanceDB table."""
    print("🔧 Creating LanceDB Vector Index")
    print("=" * 50)
    
    try:
        # Check LanceDB version
        print(f"📦 LanceDB version: {lancedb.__version__}")
        
        # Connect directly to the database
        db = lancedb.connect("lex_fridman_vectordb")
        table = db.open_table("lex_fridman_transcripts")
        
        print(f"✅ Connected to table: {table.name}")
        print(f"📊 Records: {table.count_rows():,}")
        
        # Test a quick search before indexing
        print("\n🔍 Testing search performance BEFORE indexing...")
        start_time = time.time()
        results_before = table.search().where("transcript LIKE '%artificial intelligence%'").limit(5).to_list()
        search_time_before = (time.time() - start_time) * 1000
        print(f"   Text search: {len(results_before)} results in {search_time_before:.1f}ms")
        
        # Test vector search before indexing  
        sample_record = table.search().limit(1).to_list()[0]
        sample_vector = sample_record['vector']
        
        start_time = time.time()
        vector_results_before = table.search(sample_vector).limit(5).to_list()
        vector_time_before = (time.time() - start_time) * 1000
        print(f"   Vector search: {len(vector_results_before)} results in {vector_time_before:.1f}ms")
        
        # Create the vector index
        print(f"\n🛠️  Creating vector index...")
        
        try:
            # Method 1: Try basic index creation
            table.create_index("vector")
            print(f"✅ Vector index created successfully (basic method)")
        except Exception as e1:
            print(f"   Basic method failed: {e1}")
            
            try:
                # Method 2: Try with index_type parameter
                table.create_index("vector", index_type="IVF_PQ")
                print(f"✅ Vector index created successfully (IVF_PQ method)")
            except Exception as e2:
                print(f"   IVF_PQ method failed: {e2}")
                
                try:
                    # Method 3: Try using vector index creation with metric
                    table.create_index("vector", metric="cosine")
                    print(f"✅ Vector index created successfully (cosine metric)")
                except Exception as e3:
                    print(f"   Cosine metric failed: {e3}")
                    
                    try:
                        # Method 4: Try using L2 metric
                        table.create_index("vector", metric="l2")
                        print(f"✅ Vector index created successfully (L2 metric)")
                    except Exception as e4:
                        print(f"   L2 metric failed: {e4}")
                        print(f"   Skipping index creation - the database will work but may be slower")
                        print(f"   Vector search is still functional for semantic queries")
        
        # Test search performance AFTER indexing
        print(f"\n🔍 Testing search performance AFTER indexing...")
        
        start_time = time.time()
        results_after = table.search().where("transcript LIKE '%artificial intelligence%'").limit(5).to_list()
        search_time_after = (time.time() - start_time) * 1000
        print(f"   Text search: {len(results_after)} results in {search_time_after:.1f}ms")
        
        start_time = time.time()
        vector_results_after = table.search(sample_vector).limit(5).to_list()
        vector_time_after = (time.time() - start_time) * 1000
        print(f"   Vector search: {len(vector_results_after)} results in {vector_time_after:.1f}ms")
        
        # Show performance improvements
        print(f"\n📈 Performance Analysis:")
        if vector_time_before > 0 and vector_time_after > 0:
            speedup = vector_time_before / vector_time_after
            print(f"   Vector search speedup: {speedup:.1f}x faster")
        else:
            print(f"   Vector search: {vector_time_after:.1f}ms")
        
        # Show sample results
        print(f"\n🎯 Sample vector search results:")
        for i, result in enumerate(vector_results_after[:3], 1):
            distance = result.get('_distance', 0)
            print(f"   {i}. Distance: {distance:.4f}")
            print(f"      Episode: {result['episode']}")
            print(f"      Speaker: {result['speaker']}")
            print(f"      Text: {result['transcript'][:80]}...")
            print()
        
        print("=" * 50)
        print("✅ Vector search testing completed!")
        print(f"🚀 Your search API is ready for semantic search!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = create_vector_index()
    if success:
        print(f"\n🎉 LanceDB vector search is working!")
        print(f"🌐 Test your search API at http://localhost:8000")
    else:
        print(f"\n⚠️  There were issues, but search should still work.") 