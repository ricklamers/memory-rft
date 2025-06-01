import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import lancedb
import pyarrow as pa
from tqdm import tqdm

class LexFridmanDBIngester:
    def __init__(self, db_path: str = "lex_fridman_vectordb"):
        """
        Initialize the LanceDB ingester.
        
        Args:
            db_path: Path where the LanceDB database will be created
        """
        self.db_path = db_path
        self.db = None
        self.table = None
        
    def connect_db(self):
        """Connect to or create the LanceDB database."""
        print(f"🔗 Connecting to LanceDB at: {self.db_path}")
        self.db = lancedb.connect(self.db_path)
        print(f"✅ Connected to database")
        
    def create_schema(self) -> pa.Schema:
        """
        Create the schema for the vector database table.
        
        Returns:
            PyArrow schema for the table
        """
        schema = pa.schema([
            pa.field("id", pa.string()),                    # Unique identifier: episode_chunk_id
            pa.field("episode", pa.string()),               # Episode name
            pa.field("speaker", pa.string()),               # Speaker name
            pa.field("transcript", pa.string()),            # Transcript text
            pa.field("time", pa.string()),                  # Timestamp
            pa.field("chunk_id", pa.int64()),               # Chunk ID within episode
            pa.field("embedding_model", pa.string()),       # Model used for embedding
            pa.field("embedding_dimension", pa.int64()),    # Embedding dimension
            pa.field("vector", pa.list_(pa.float32())),     # Embedding vector
        ])
        return schema
        
    def load_embedding_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load embeddings from a JSON file and convert to database records.
        
        Args:
            filepath: Path to the embedding JSON file
            
        Returns:
            List of records ready for database insertion
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = []
            episode_name = data['episode_name']
            
            for chunk in data['chunks']:
                # Create a unique ID combining episode and chunk ID
                unique_id = f"{episode_name}_{chunk['chunk_id']}"
                
                record = {
                    'id': unique_id,
                    'episode': chunk['episode'],
                    'speaker': chunk['speaker'],
                    'transcript': chunk['transcript'],
                    'time': chunk['time'],
                    'chunk_id': chunk['chunk_id'],
                    'embedding_model': chunk['embedding_model'],
                    'embedding_dimension': chunk['embedding_dimension'],
                    'vector': chunk['embedding']  # LanceDB expects this to be called 'vector'
                }
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return []
    
    def create_table(self, table_name: str = "lex_fridman_transcripts", sample_data: List[Dict] = None):
        """
        Create the vector database table.
        
        Args:
            table_name: Name of the table to create
            sample_data: Sample data to use for table creation
        """
        # Check if table already exists
        try:
            existing_tables = self.db.table_names()
            if table_name in existing_tables:
                print(f"📋 Table '{table_name}' already exists, using existing table")
                self.table = self.db.open_table(table_name)
                return
        except:
            pass
        
        print(f"🆕 Creating new table: {table_name}")
        
        if sample_data and len(sample_data) > 0:
            # Create table with sample data
            self.table = self.db.create_table(table_name, data=sample_data)
            print(f"✅ Table '{table_name}' created with {len(sample_data)} initial records")
        else:
            # Create empty table - we'll add data later
            schema = self.create_schema()
            empty_records = [{
                'id': 'temp_id',
                'episode': 'temp_episode',
                'speaker': 'temp_speaker',
                'transcript': 'temp_transcript',
                'time': '00:00:00',
                'chunk_id': 0,
                'embedding_model': 'temp_model',
                'embedding_dimension': 1536,
                'vector': [0.0] * 1536
            }]
            self.table = self.db.create_table(table_name, data=empty_records)
            # Delete the temporary record
            self.table.delete("id = 'temp_id'")
            print(f"✅ Empty table '{table_name}' created")
    
    def ingest_embeddings(self, embeddings_directory: str, batch_size: int = 1000):
        """
        Ingest all embedding files from the directory into LanceDB.
        
        Args:
            embeddings_directory: Directory containing embedding JSON files
            batch_size: Number of records to insert in each batch
        """
        if not os.path.exists(embeddings_directory):
            print(f"❌ Embeddings directory '{embeddings_directory}' not found!")
            return
        
        # Find all embedding files
        embedding_files = []
        for filename in os.listdir(embeddings_directory):
            if filename.endswith('_embeddings.json') and filename != 'embeddings_summary.json':
                embedding_files.append(os.path.join(embeddings_directory, filename))
        
        if not embedding_files:
            print(f"❌ No embedding files found in {embeddings_directory}")
            return
        
        print(f"📁 Found {len(embedding_files)} embedding files to ingest")
        print(f"📦 Batch size: {batch_size}")
        print()
        
        total_records = 0
        successful_files = 0
        failed_files = []
        table_created = False
        
        start_time = time.time()
        
        # Process each embedding file
        for i, filepath in enumerate(embedding_files):
            filename = os.path.basename(filepath)
            print(f"\n[{i+1}/{len(embedding_files)}] Processing {filename}...")
            
            # Load records from file
            records = self.load_embedding_file(filepath)
            
            if not records:
                failed_files.append(filename)
                print(f"  ❌ No records loaded from {filename}")
                continue
            
            # Create table with first batch of records if not created yet
            if not table_created:
                first_batch = records[:min(batch_size, len(records))]
                self.create_table("lex_fridman_transcripts", sample_data=first_batch)
                table_created = True
                total_records += len(first_batch)
                print(f"  ✅ Table created and {len(first_batch)} initial records inserted")
                
                # Continue with remaining records from this file
                remaining_records = records[len(first_batch):]
            else:
                remaining_records = records
            
            try:
                # Insert remaining records in batches
                for batch_start in range(0, len(remaining_records), batch_size):
                    batch_end = min(batch_start + batch_size, len(remaining_records))
                    batch_records = remaining_records[batch_start:batch_end]
                    
                    if batch_records:  # Only insert if there are records
                        print(f"  📝 Inserting batch: {len(batch_records)} records...")
                        self.table.add(batch_records)
                        total_records += len(batch_records)
                
                successful_files += 1
                print(f"  ✅ Successfully processed {len(records)} records from {filename}")
                
            except Exception as e:
                failed_files.append(filename)
                print(f"  ❌ Error inserting records from {filename}: {e}")
        
        total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 LANCEDB INGESTION SUMMARY")
        print("=" * 60)
        print(f"📁 Total files processed: {len(embedding_files)}")
        print(f"✅ Successful ingestions: {successful_files}")
        print(f"❌ Failed ingestions: {len(failed_files)}")
        print(f"📝 Total records inserted: {total_records:,}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🚀 Processing speed: {total_records/(total_time/60):.1f} records/minute")
        print(f"💾 Database location: {self.db_path}")
        
        if failed_files:
            print(f"\n❌ Failed files:")
            for failed_file in failed_files:
                print(f"   - {failed_file}")
        
        # Show table info
        if self.table:
            try:
                count = self.table.count_rows()
                print(f"\n📊 Final table statistics:")
                print(f"   - Total rows in database: {count:,}")
                print(f"   - Table name: {self.table.name}")
            except Exception as e:
                print(f"   - Could not get table statistics: {e}")
    
    def create_index(self, index_type: str = "IVF_PQ"):
        """
        Create a vector index for faster similarity search.
        
        Args:
            index_type: Type of index to create (IVF_PQ, IVF_FLAT, etc.)
        """
        if not self.table:
            print("❌ No table available to index")
            return
        
        try:
            print(f"🔍 Creating {index_type} index on vector column...")
            
            # LanceDB v0.22+ API - create index on the vector column
            if index_type == "IVF_PQ":
                self.table.create_index(
                    "vector",  # Column name as first parameter
                    config={
                        "index_type": "IVF_PQ",
                        "num_partitions": 256,
                        "num_sub_vectors": 96
                    }
                )
            elif index_type == "IVF_FLAT":
                self.table.create_index(
                    "vector",
                    config={
                        "index_type": "IVF_FLAT", 
                        "num_partitions": 256
                    }
                )
            else:
                # Default to IVF_PQ
                self.table.create_index("vector")
            
            print(f"✅ Index created successfully")
            
        except Exception as e:
            print(f"⚠️  Could not create index: {e}")
            print("   Trying alternative index creation method...")
            
            # Try alternative method for older LanceDB versions
            try:
                self.table.create_index("vector", replace=True)
                print(f"✅ Index created with alternative method")
            except Exception as e2:
                print(f"⚠️  Alternative index creation also failed: {e2}")
                print("   Database will still work but searches may be slower")
                print("   Consider updating LanceDB or checking the documentation")
    
    def test_search(self, query_text: str = "artificial intelligence", limit: int = 5):
        """
        Test the database with a sample search.
        
        Args:
            query_text: Text to search for
            limit: Number of results to return
        """
        if not self.table:
            print("❌ No table available for testing")
            return
        
        print(f"\n🔍 Testing search for: '{query_text}'")
        print(f"🎯 Looking for top {limit} results...")
        
        try:
            # For testing, we'll do a simple text search
            # In practice, you'd want to embed the query text first
            results = (self.table
                      .search()
                      .where(f"transcript LIKE '%{query_text}%'")
                      .limit(limit)
                      .to_pandas())
            
            if len(results) > 0:
                print(f"✅ Found {len(results)} results:")
                for i, row in results.iterrows():
                    print(f"\n  {i+1}. Episode: {row['episode']}")
                    print(f"     Speaker: {row['speaker']}")
                    print(f"     Time: {row['time']}")
                    print(f"     Text: {row['transcript'][:100]}...")
            else:
                print(f"❌ No results found for '{query_text}'")
                
        except Exception as e:
            print(f"❌ Error during test search: {e}")

def main():
    """Main function to run the ingestion process."""
    
    # Configuration
    embeddings_dir = "lex_fridman_embeddings"
    db_path = "lex_fridman_vectordb"
    table_name = "lex_fridman_transcripts"
    
    print("🚀 Starting LanceDB ingestion for Lex Fridman transcripts...")
    print(f"📂 Source directory: {embeddings_dir}")
    print(f"💾 Target database: {db_path}")
    print(f"📋 Table name: {table_name}")
    print()
    
    # Check if embeddings directory exists
    if not os.path.exists(embeddings_dir):
        print(f"❌ Embeddings directory '{embeddings_dir}' not found!")
        print("Please run the embedding generation script first.")
        return
    
    # Create ingester and run the process
    ingester = LexFridmanDBIngester(db_path)
    
    try:
        # Connect to database
        ingester.connect_db()
        
        # Ingest all embeddings (table creation is handled automatically)
        ingester.ingest_embeddings(embeddings_dir)
        
        # Create index for faster searches
        if ingester.table:
            ingester.create_index()
        
        # Test the database
        if ingester.table:
            ingester.test_search("artificial intelligence")
        
        print(f"\n🎉 Ingestion completed successfully!")
        print(f"💾 Database ready at: {db_path}")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return

if __name__ == "__main__":
    main() 