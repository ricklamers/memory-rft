#!/usr/bin/env python3
"""
Ingest QA pairs into LanceDB for semantic search.
"""

import json
import os
import lancedb
import numpy as np
from typing import List, Dict
import openai
from datetime import datetime
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Create embedding for a single text using OpenAI API."""
    try:
        response = openai.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

def create_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Create embeddings for multiple texts in a single API call."""
    try:
        response = openai.embeddings.create(
            input=texts,
            model=model
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        return [None] * len(texts)

def process_qa_file(qa_file_path: str) -> List[Dict]:
    """Process a QA file and prepare records for LanceDB."""
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    records = []
    
    for i, qa_pair in enumerate(qa_data["qa_pairs"]):
        # Create record for the question
        question_record = {
            "type": "qa_question",
            "episode": qa_data["episode"],
            "episode_file": qa_data["episode_file"],
            "qa_index": i,
            "text": qa_pair["question"],
            "answer": qa_pair["answer"],
            "difficulty": qa_pair.get("difficulty", "hard"),
            "requires_synthesis": qa_pair.get("requires_synthesis", True),
            "created_at": qa_data["generated_at"]
        }
        records.append(question_record)
        
        # Create record for the answer
        answer_record = {
            "type": "qa_answer",
            "episode": qa_data["episode"],
            "episode_file": qa_data["episode_file"],
            "qa_index": i,
            "text": qa_pair["answer"],
            "question": qa_pair["question"],
            "difficulty": qa_pair.get("difficulty", "hard"),
            "requires_synthesis": qa_pair.get("requires_synthesis", True),
            "created_at": qa_data["generated_at"]
        }
        records.append(answer_record)
    
    return records

def main():
    parser = argparse.ArgumentParser(description="Ingest QA pairs into LanceDB")
    parser.add_argument("--qa-dir", default="dataset/lex_fridman_qa_pairs",
                       help="Directory containing QA JSON files")
    parser.add_argument("--db-path", default="dataset/lex_fridman_vectordb",
                       help="Path to LanceDB database")
    parser.add_argument("--table-name", default="lex_fridman_qa",
                       help="Name of the LanceDB table for QA pairs")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for embedding generation (max 1024 for OpenAI)")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                       help="OpenAI embedding model to use")
    
    args = parser.parse_args()
    
    # Connect to LanceDB
    db = lancedb.connect(args.db_path)
    
    # Get all QA files
    qa_files = [f for f in os.listdir(args.qa_dir) if f.endswith('_qa.json')]
    qa_files.sort()
    
    print(f"\nProcessing {len(qa_files)} QA files")
    print(f"Database path: {args.db_path}")
    print(f"Table name: {args.table_name}")
    print(f"Embedding model: {args.embedding_model}")
    print("-" * 80)
    
    # Process all QA files and collect records
    all_records = []
    
    for qa_file in tqdm(qa_files, desc="Loading QA files"):
        qa_file_path = os.path.join(args.qa_dir, qa_file)
        records = process_qa_file(qa_file_path)
        all_records.extend(records)
    
    print(f"\nTotal QA records to process: {len(all_records)}")
    
    # Generate embeddings in batches
    print("\nGenerating embeddings...")
    for i in tqdm(range(0, len(all_records), args.batch_size), desc="Embedding batches"):
        batch = all_records[i:i + args.batch_size]
        texts = [record["text"] for record in batch]
        
        embeddings = create_embeddings_batch(texts, args.embedding_model)
        
        for j, embedding in enumerate(embeddings):
            if embedding:
                batch[j]["vector"] = embedding
            else:
                # Skip records with failed embeddings
                batch[j]["vector"] = None
        
        # Small delay to respect rate limits
        time.sleep(0.1)
    
    # Filter out records without embeddings
    valid_records = [r for r in all_records if r.get("vector") is not None]
    print(f"\nSuccessfully embedded {len(valid_records)} out of {len(all_records)} records")
    
    # Create or append to table
    table_names = db.table_names()
    if args.table_name in table_names:
        print(f"\nAppending to existing table '{args.table_name}'...")
        table = db.open_table(args.table_name)
        table.add(valid_records)
    else:
        print(f"\nCreating new table '{args.table_name}'...")
        table = db.create_table(args.table_name, data=valid_records)
    
    # Create indices for better query performance
    print("\nCreating indices...")
    table.create_index(num_partitions=256, num_sub_vectors=96)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total QA files processed: {len(qa_files)}")
    print(f"Total records created: {len(all_records)}")
    print(f"Total records with embeddings: {len(valid_records)}")
    print(f"Questions: {len([r for r in valid_records if r['type'] == 'qa_question'])}")
    print(f"Answers: {len([r for r in valid_records if r['type'] == 'qa_answer'])}")
    
    # Save ingestion summary
    summary_data = {
        "ingested_at": datetime.now().isoformat(),
        "qa_files_processed": len(qa_files),
        "total_records": len(all_records),
        "embedded_records": len(valid_records),
        "embedding_model": args.embedding_model,
        "table_name": args.table_name,
        "db_path": args.db_path
    }
    
    summary_path = os.path.join(args.qa_dir, "ingestion_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nIngestion summary saved to: {summary_path}")
    
    # Example search
    print("\n" + "=" * 80)
    print("EXAMPLE SEARCH")
    print("=" * 80)
    
    query = "How do different topics and themes connect throughout the conversation?"
    results = table.search(create_embedding(query, args.embedding_model)).limit(5).to_pandas()
    
    print(f"Query: {query}\n")
    for i, row in results.iterrows():
        print(f"{i+1}. Type: {row['type']}")
        print(f"   Episode: {row['episode_file']}")
        print(f"   Text: {row['text'][:200]}...")
        print(f"   Distance: {row['_distance']:.4f}")
        print()

if __name__ == "__main__":
    main() 