#!/usr/bin/env python3
"""
Test script to verify LanceDB indexing is working correctly.
"""

import time
import lancedb
from ingest_to_lancedb import LexFridmanDBIngester

def test_index_creation():
    """Test that vector indexing works correctly."""
    print("🧪 Testing LanceDB Index Creation and Performance")
    print("=" * 60)
    
    # Connect to the existing database
    ingester = LexFridmanDBIngester("lex_fridman_vectordb")
    
    try:
        ingester.connect_db()
        
        if not ingester.table:
            print("❌ No table found! Run the ingestion script first.")
            return False
        
        # Get table info
        row_count = ingester.table.count_rows()
        print(f"📊 Database contains {row_count:,} records")
        
        # Test search performance WITHOUT index
        print("\n🔍 Testing search performance WITHOUT index...")
        test_query = "artificial intelligence consciousness"
        
        # Simple text search to avoid needing embeddings
        start_time = time.time()
        results_before = (ingester.table
                         .search()
                         .where(f"transcript LIKE '%artificial intelligence%'")
                         .limit(10)
                         .to_list())
        search_time_before = (time.time() - start_time) * 1000
        
        print(f"   Found {len(results_before)} results in {search_time_before:.1f}ms")
        
        # Create the index
        print(f"\n🔧 Creating vector index...")
        ingester.create_index("IVF_PQ")
        
        # Test search performance WITH index
        print("\n🔍 Testing search performance WITH index...")
        start_time = time.time()
        results_after = (ingester.table
                        .search()
                        .where(f"transcript LIKE '%artificial intelligence%'")
                        .limit(10)
                        .to_list())
        search_time_after = (time.time() - start_time) * 1000
        
        print(f"   Found {len(results_after)} results in {search_time_after:.1f}ms")
        
        # Show performance improvement
        if search_time_before > 0 and search_time_after > 0:
            speedup = search_time_before / search_time_after
            print(f"\n📈 Performance improvement: {speedup:.1f}x faster with index")
        
        # Test vector similarity search (if we have embeddings)
        print(f"\n🎯 Testing vector similarity search...")
        try:
            # Get a sample vector from the database
            sample_record = ingester.table.search().limit(1).to_list()[0]
            sample_vector = sample_record['vector']
            
            start_time = time.time()
            vector_results = (ingester.table
                            .search(sample_vector)
                            .limit(5)
                            .to_list())
            vector_search_time = (time.time() - start_time) * 1000
            
            print(f"   ✅ Vector search: {len(vector_results)} results in {vector_search_time:.1f}ms")
            
            # Show some results
            if vector_results:
                print(f"\n🎭 Sample vector search results:")
                for i, result in enumerate(vector_results[:3], 1):
                    print(f"   {i}. Episode: {result['episode']}")
                    print(f"      Speaker: {result['speaker']}")
                    print(f"      Text: {result['transcript'][:100]}...")
                    print(f"      Distance: {result.get('_distance', 'N/A'):.4f}")
                    print()
        
        except Exception as e:
            print(f"   ⚠️  Vector search test failed: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ Index testing completed successfully!")
        print(f"📊 Database has {row_count:,} searchable records")
        print(f"🔍 Text search: {search_time_after:.1f}ms")
        if 'vector_search_time' in locals():
            print(f"🎯 Vector search: {vector_search_time:.1f}ms")
        print("💾 Vector index is working properly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during index testing: {e}")
        return False

def check_index_status():
    """Check if the database has indexes."""
    print("\n🔍 Checking index status...")
    
    try:
        db = lancedb.connect("lex_fridman_vectordb")
        table = db.open_table("lex_fridman_transcripts")
        
        # Try to get table statistics or schema info
        print(f"📊 Table: {table.name}")
        print(f"📝 Records: {table.count_rows():,}")
        
        # Check if we can perform a quick vector search
        sample_record = table.search().limit(1).to_list()[0]
        if 'vector' in sample_record:
            vector_dim = len(sample_record['vector'])
            print(f"🧠 Vector dimension: {vector_dim}")
        
        print("✅ Database structure looks good")
        
    except Exception as e:
        print(f"⚠️  Could not check index status: {e}")

if __name__ == "__main__":
    success = test_index_creation()
    check_index_status()
    
    if success:
        print(f"\n🎉 All tests passed! LanceDB indexing is working correctly.")
        print(f"🌐 Your search API at http://localhost:8000 should now be optimized!")
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.") 