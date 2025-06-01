#!/usr/bin/env python3
"""
Generate QA pairs from Lex Fridman transcripts using Google Gemini 2.5 Pro.
QA pairs are designed to require information from multiple parts of the transcript.
"""

import json
import os
import time
import logging
from typing import List, Dict, Tuple
from datetime import datetime
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import argparse
import lancedb
import openai
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure OpenAI for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
def setup_logging(log_dir: str = "dataset/lex_fridman_qa_pairs/logs"):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('qa_generation')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler for all logs
    log_file = os.path.join(log_dir, f'qa_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Stream handler for model output
    stream_log_file = os.path.join(log_dir, f'model_stream_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    stream_handler = logging.FileHandler(stream_log_file, encoding='utf-8')
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Create a separate logger for streaming
    stream_logger = logging.getLogger('model_stream')
    stream_logger.setLevel(logging.DEBUG)
    stream_logger.addHandler(stream_handler)
    stream_logger.addHandler(console_handler)
    
    return logger, stream_logger, log_file, stream_log_file

# Get loggers
logger, stream_logger, _, _ = setup_logging()

def preprocess_transcript(transcript_data: Dict) -> str:
    """
    Preprocess transcript to minimal format:
    - Remove JSON structure
    - Remove timestamps
    - Keep only speaker and text
    """
    lines = []
    for entry in transcript_data["content"]:
        speaker = entry["speaker"]
        text = entry["transcript"]
        lines.append(f"{speaker}: {text}")
    
    return "\n\n".join(lines)

def generate_qa_pairs(transcript_text: str, episode_title: str, num_pairs: int = 20) -> List[Dict]:
    """
    Generate QA pairs using Gemini 2.5 Pro with streaming output.
    Questions should require multiple parts of the transcript to answer.
    """
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = f"""You are an expert at creating challenging question-answer pairs from podcast transcripts. 

Episode: {episode_title}

IMPORTANT REQUIREMENTS:
1. Generate exactly {num_pairs} question-answer pairs from this transcript
2. Each question MUST require information from MULTIPLE different parts of the transcript to answer properly
3. Questions should be complex and require synthesis of ideas discussed at different points
4. Answers should be comprehensive, citing relevant parts from throughout the transcript
5. Focus on deeper themes, connections between topics, and evolution of ideas throughout the conversation
6. Include questions about:
   - How topics relate to each other
   - How perspectives change throughout the conversation
   - Synthesis of multiple viewpoints discussed
   - Connections between personal anecdotes and broader themes
   - Overall narrative arc of the conversation

Return the response in valid JSON format with this structure:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "...",
      "difficulty": "hard",
      "requires_synthesis": true
    }}
  ]
}}

TRANSCRIPT:
{transcript_text}
"""
    
    try:
        logger.info(f"Requesting {num_pairs} QA pairs from Gemini...")
        stream_logger.info(f"\n{'='*80}\nSTREAMING OUTPUT FOR: {episode_title}\n{'='*80}\n")
        
        # Generate with streaming
        response = model.generate_content(prompt, stream=True)
        
        # Collect the full response while streaming
        full_response = ""
        token_count = 0
        
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                token_count += len(chunk.text.split())  # Rough token estimate
                
                # Log the streaming chunk
                stream_logger.debug(chunk.text)
                
                # Log progress every 100 "tokens"
                if token_count % 100 == 0:
                    logger.debug(f"Streamed approximately {token_count} tokens...")
        
        stream_logger.info(f"\n{'='*80}\nEND OF STREAMING\n{'='*80}\n")
        logger.info(f"Received complete response (~{token_count} tokens)")
        
        # Parse the JSON response
        result_text = full_response
        
        # Clean up the response if needed (remove markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        qa_data = json.loads(result_text.strip())
        logger.info(f"Successfully parsed {len(qa_data['qa_pairs'])} QA pairs")
        
        return qa_data["qa_pairs"]
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.debug(f"Raw response: {full_response[:500]}...")
        return []
    except Exception as e:
        logger.error(f"Error generating QA pairs: {e}")
        return []

def process_episode(transcript_path: str, output_dir: str, num_pairs: int = 20) -> Tuple[str, bool, int]:
    """Process a single episode to generate QA pairs."""
    episode_name = os.path.basename(transcript_path).replace('.json', '')
    logger.info(f"Processing episode: {episode_name}")
    
    try:
        # Load transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Preprocess transcript
        logger.info(f"Preprocessing transcript...")
        transcript_text = preprocess_transcript(transcript_data)
        
        # Check token limit (roughly 4 chars per token)
        char_count = len(transcript_text)
        estimated_tokens = char_count / 4
        logger.info(f"Transcript length: {char_count:,} chars (~{estimated_tokens:,.0f} tokens)")
        
        # Generate QA pairs
        logger.info(f"Generating {num_pairs} QA pairs...")
        qa_pairs = generate_qa_pairs(transcript_text, transcript_data["episode"], num_pairs)
        
        if not qa_pairs:
            logger.warning(f"No QA pairs generated for {episode_name}")
            return episode_name, False, 0
        
        # Save QA pairs
        output_data = {
            "episode": transcript_data["episode"],
            "episode_file": episode_name,
            "speakers": transcript_data["speakers"],
            "num_qa_pairs": len(qa_pairs),
            "generated_at": datetime.now().isoformat(),
            "qa_pairs": qa_pairs
        }
        
        output_path = os.path.join(output_dir, f"{episode_name}_qa.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Generated {len(qa_pairs)} QA pairs for {episode_name}")
        return episode_name, True, len(qa_pairs)
    
    except Exception as e:
        logger.error(f"Error processing {episode_name}: {e}", exc_info=True)
        return episode_name, False, 0

def get_already_processed_episodes(output_dir: str) -> Dict[str, int]:
    """Get list of episodes that already have QA pairs generated."""
    processed = {}
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('_qa.json'):
                try:
                    with open(os.path.join(output_dir, file), 'r') as f:
                        data = json.load(f)
                        episode_name = file.replace('_qa.json', '')
                        processed[episode_name] = data.get('num_qa_pairs', 0)
                except:
                    pass
    return processed

# LanceDB ingestion functions
def create_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Create embedding for a single text using OpenAI API."""
    try:
        response = openai.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
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
        logger.error(f"Error creating batch embeddings: {e}")
        return [None] * len(texts)

def process_qa_file_for_db(qa_file_path: str) -> List[Dict]:
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

def ingest_to_lancedb(qa_dir: str, db_path: str, table_name: str = "lex_fridman_qa", 
                     batch_size: int = 1024, embedding_model: str = "text-embedding-3-small"):
    """Ingest QA pairs into LanceDB."""
    logger.info("="*80)
    logger.info("Starting LanceDB ingestion")
    logger.info("="*80)
    
    # Connect to LanceDB
    db = lancedb.connect(db_path)
    
    # Get all QA files
    qa_files = [f for f in os.listdir(qa_dir) if f.endswith('_qa.json')]
    qa_files.sort()
    
    logger.info(f"Processing {len(qa_files)} QA files for LanceDB")
    
    # Process all QA files and collect records
    all_records = []
    
    for qa_file in tqdm(qa_files, desc="Loading QA files"):
        qa_file_path = os.path.join(qa_dir, qa_file)
        records = process_qa_file_for_db(qa_file_path)
        all_records.extend(records)
    
    logger.info(f"Total QA records to process: {len(all_records)}")
    
    # Generate embeddings in batches
    logger.info("Generating embeddings...")
    for i in tqdm(range(0, len(all_records), batch_size), desc="Embedding batches"):
        batch = all_records[i:i + batch_size]
        texts = [record["text"] for record in batch]
        
        embeddings = create_embeddings_batch(texts, embedding_model)
        
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
    logger.info(f"Successfully embedded {len(valid_records)} out of {len(all_records)} records")
    
    # Create or append to table
    table_names = db.table_names()
    if table_name in table_names:
        logger.info(f"Appending to existing table '{table_name}'...")
        table = db.open_table(table_name)
        table.add(valid_records)
    else:
        logger.info(f"Creating new table '{table_name}'...")
        table = db.create_table(table_name, data=valid_records)
    
    # Create indices for better query performance
    logger.info("Creating indices...")
    table.create_index(num_partitions=256, num_sub_vectors=96)
    
    logger.info("LanceDB ingestion completed successfully!")
    
    return len(valid_records)

def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from Lex Fridman transcripts")
    parser.add_argument("--input-dir", default="dataset/lex_fridman_transcripts", 
                       help="Directory containing transcript JSON files")
    parser.add_argument("--output-dir", default="dataset/lex_fridman_qa_pairs",
                       help="Directory to save QA pairs")
    parser.add_argument("--num-pairs", type=int, default=20,
                       help="Number of QA pairs to generate per episode")
    parser.add_argument("--episodes", nargs="+",
                       help="Specific episodes to process (default: all)")
    parser.add_argument("--max-episodes", type=int,
                       help="Maximum number of episodes to process")
    parser.add_argument("--continue", dest="continue_processing", action="store_true",
                       help="Continue/fill gaps (process only episodes without QA pairs)")
    parser.add_argument("--no-ingest", action="store_true",
                       help="Skip LanceDB ingestion after QA generation")
    parser.add_argument("--db-path", default="dataset/lex_fridman_vectordb",
                       help="Path to LanceDB database")
    parser.add_argument("--table-name", default="lex_fridman_qa",
                       help="Name of the LanceDB table for QA pairs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging with correct output directory
    logger, stream_logger, log_file, stream_log_file = setup_logging(
        os.path.join(args.output_dir, "logs")
    )
    
    logger.info(f"QA Generation started at {datetime.now()}")
    logger.info(f"Main log file: {log_file}")
    logger.info(f"Stream log file: {stream_log_file}")
    
    # Get all available transcript files
    all_transcript_files = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.json') and file != 'scraping_summary.json':
            all_transcript_files.append(file)
    
    # Check for already processed episodes if continuing
    already_processed = {}
    gaps_found = []
    if args.continue_processing:
        already_processed = get_already_processed_episodes(args.output_dir)
        
        # Find gaps - episodes that have transcripts but no QA pairs
        for transcript_file in all_transcript_files:
            episode_name = transcript_file.replace('.json', '')
            if episode_name not in already_processed:
                gaps_found.append(episode_name)
        
        if already_processed:
            logger.info(f"Found {len(already_processed)} already processed episodes")
            for ep, count in list(already_processed.items())[:5]:
                logger.info(f"  - {ep}: {count} QA pairs")
            if len(already_processed) > 5:
                logger.info(f"  ... and {len(already_processed) - 5} more")
        
        if gaps_found:
            logger.info(f"Found {len(gaps_found)} gaps to fill:")
            for gap in gaps_found[:10]:
                logger.info(f"  - {gap} (missing QA pairs)")
            if len(gaps_found) > 10:
                logger.info(f"  ... and {len(gaps_found) - 10} more")
    
    # Get transcript files to process
    transcript_files = []
    for file in all_transcript_files:
        episode_name = file.replace('.json', '')
        
        # Skip if already processed and continuing (unless it's a gap)
        if args.continue_processing and episode_name in already_processed:
            continue
        
        if args.episodes:
            # Check if this episode is in the requested list
            if any(ep in episode_name for ep in args.episodes):
                transcript_files.append(os.path.join(args.input_dir, file))
        else:
            transcript_files.append(os.path.join(args.input_dir, file))
    
    # Sort files
    transcript_files.sort()
    
    # Limit number of episodes if specified
    if args.max_episodes:
        transcript_files = transcript_files[:args.max_episodes]
    
    print(f"\nGenerating QA pairs for {len(transcript_files)} episodes")
    if args.continue_processing:
        if already_processed:
            print(f"Already processed: {len(already_processed)} episodes")
        if gaps_found:
            print(f"Filling gaps: {len([f for f in transcript_files if os.path.basename(f).replace('.json', '') in gaps_found])} episodes")
    print(f"Output directory: {args.output_dir}")
    print(f"QA pairs per episode: {args.num_pairs}")
    print(f"Logs directory: {os.path.join(args.output_dir, 'logs')}")
    print("-" * 80)
    
    # Process each episode
    results = []
    total_qa_pairs = sum(already_processed.values()) if args.continue_processing else 0
    successful = len(already_processed) if args.continue_processing else 0
    newly_processed = 0
    
    for i, transcript_path in enumerate(transcript_files, 1):
        episode_name = os.path.basename(transcript_path).replace('.json', '')
        is_gap = episode_name in gaps_found if args.continue_processing else False
        
        print(f"\n[{i}/{len(transcript_files)}] Processing: {episode_name}" + (" (filling gap)" if is_gap else ""))
        
        name, success, num_pairs = process_episode(transcript_path, args.output_dir, args.num_pairs)
        results.append((name, success, num_pairs))
        
        if success:
            newly_processed += 1
            successful += 1
            total_qa_pairs += num_pairs
        
        # Rate limiting (Gemini has generous limits but let's be respectful)
        if i < len(transcript_files):
            time.sleep(2)  # 2 second delay between requests
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_episodes = len(all_transcript_files)
    failed = len(transcript_files) - newly_processed
    
    print(f"Total transcript files available: {total_episodes}")
    print(f"Total with QA pairs: {successful}")
    print(f"Newly processed this run: {newly_processed}")
    if args.continue_processing and gaps_found:
        gaps_filled = len([r for r in results if r[0] in gaps_found and r[1]])
        print(f"Gaps filled: {gaps_filled}")
    print(f"Failed: {failed}")
    print(f"Total QA pairs: {total_qa_pairs}")
    print(f"Average QA pairs per episode: {total_qa_pairs / successful:.1f}" if successful > 0 else "N/A")
    
    logger.info("="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total transcript files available: {total_episodes}")
    logger.info(f"Total with QA pairs: {successful}")
    logger.info(f"Newly processed: {newly_processed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total QA pairs: {total_qa_pairs}")
    
    # Save summary
    summary_data = {
        "generated_at": datetime.now().isoformat(),
        "total_transcripts_available": total_episodes,
        "total_with_qa_pairs": successful,
        "newly_processed": newly_processed,
        "failed": failed,
        "total_qa_pairs": total_qa_pairs,
        "results": [{"episode": name, "success": success, "qa_pairs": num} for name, success, num in results],
        "continued_from_previous": args.continue_processing,
        "already_processed_count": len(already_processed),
        "gaps_found": len(gaps_found) if args.continue_processing else 0,
        "gaps_filled": len([r for r in results if r[0] in gaps_found and r[1]]) if args.continue_processing else 0
    }
    
    summary_path = os.path.join(args.output_dir, "generation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"Logs saved to: {os.path.join(args.output_dir, 'logs')}")
    
    # Ingest to LanceDB unless disabled
    if not args.no_ingest and newly_processed > 0:
        print("\n" + "=" * 80)
        print("Starting LanceDB ingestion...")
        print("=" * 80)
        
        try:
            num_ingested = ingest_to_lancedb(
                args.output_dir, 
                args.db_path,
                args.table_name
            )
            print(f"\n✓ Successfully ingested {num_ingested} records to LanceDB")
        except Exception as e:
            logger.error(f"Error during LanceDB ingestion: {e}", exc_info=True)
            print(f"\n✗ Error during LanceDB ingestion: {e}")
            print("You can manually run ingestion later")

if __name__ == "__main__":
    main()