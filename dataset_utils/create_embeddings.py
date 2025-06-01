import json
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ChunkEmbeddingGenerator:
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small", batch_size: int = 1024):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI embedding model to use
            batch_size: Number of texts to process in each batch
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_dim = 1536  # text-embedding-3-small dimension
        self.batch_size = batch_size
        self.rate_limit_delay = 1.0  # Delay between batch API calls
        
    def parse_chunk_file(self, filepath: str) -> List[Dict[str, str]]:
        """
        Parse a chunk file and extract individual chunks.
        
        Args:
            filepath: Path to the chunk file
            
        Returns:
            List of dictionaries containing chunk data
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by separator
            chunk_texts = content.split('\n\n---\n\n')
            
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                
                # Parse the XML-like format
                episode_match = re.search(r'<episode>(.*?)</episode>', chunk_text, re.DOTALL)
                speaker_match = re.search(r'<speaker>(.*?)</speaker>', chunk_text, re.DOTALL)
                transcript_match = re.search(r'<transcript>(.*?)</transcript>', chunk_text, re.DOTALL)
                time_match = re.search(r'<time>(.*?)</time>', chunk_text, re.DOTALL)
                
                if all([episode_match, speaker_match, transcript_match, time_match]):
                    chunks.append({
                        'chunk_id': i,
                        'episode': episode_match.group(1).strip(),
                        'speaker': speaker_match.group(1).strip(),
                        'transcript': transcript_match.group(1).strip(),
                        'time': time_match.group(1).strip(),
                        'raw_chunk': chunk_text
                    })
                else:
                    print(f"  ⚠️  Could not parse chunk {i} in {filepath}")
            
            return chunks
            
        except Exception as e:
            print(f"  ❌ Error parsing {filepath}: {e}")
            return []
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors as lists of floats
        """
        try:
            # Add a delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            # Extract embeddings in the same order as input texts
            embeddings = [data.embedding for data in response.data]
            return embeddings
            
        except Exception as e:
            print(f"  ❌ Error creating batch embeddings: {e}")
            return [None] * len(texts)
    
    def process_chunk_file(self, chunk_filepath: str, output_directory: str) -> Dict[str, Any]:
        """
        Process a single chunk file and create embeddings for all chunks using batch processing.
        
        Args:
            chunk_filepath: Path to the chunk file
            output_directory: Directory to save embedding files
            
        Returns:
            Dictionary with processing results
        """
        filename = os.path.basename(chunk_filepath)
        episode_name = filename.replace('_chunks.txt', '')
        
        print(f"Processing {filename}...")
        
        # Parse chunks from file
        chunks = self.parse_chunk_file(chunk_filepath)
        
        if not chunks:
            return {
                'episode_name': episode_name,
                'success': False,
                'error': 'No chunks found or parsing failed',
                'chunks_processed': 0
            }
        
        print(f"  📝 Found {len(chunks)} chunks, processing in batches of {self.batch_size}...")
        
        # Create embeddings in batches
        embedded_chunks = []
        successful_embeddings = 0
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_idx + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_idx:batch_end]
            batch_num = (batch_idx // self.batch_size) + 1
            
            # Extract texts for this batch
            batch_texts = [chunk['transcript'] for chunk in batch_chunks]
            
            print(f"    🔄 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)...")
            
            # Create embeddings for the batch
            batch_embeddings = self.create_embeddings_batch(batch_texts)
            
            # Process each embedding in the batch
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                if embedding is not None:
                    embedded_chunk = {
                        'chunk_id': chunk['chunk_id'],
                        'episode': chunk['episode'],
                        'speaker': chunk['speaker'],
                        'transcript': chunk['transcript'],
                        'time': chunk['time'],
                        'embedding': embedding,
                        'embedding_model': self.model,
                        'embedding_dimension': len(embedding)
                    }
                    embedded_chunks.append(embedded_chunk)
                    successful_embeddings += 1
                else:
                    print(f"      ⚠️  Failed to create embedding for chunk {chunk['chunk_id']}")
        
        if not embedded_chunks:
            return {
                'episode_name': episode_name,
                'success': False,
                'error': 'No embeddings created',
                'chunks_processed': 0
            }
        
        # Save embeddings to file
        output_filename = f"{episode_name}_embeddings.json"
        output_filepath = os.path.join(output_directory, output_filename)
        
        try:
            # Create the episode embeddings file
            episode_data = {
                'episode_name': episode_name,
                'total_chunks': len(embedded_chunks),
                'embedding_model': self.model,
                'embedding_dimension': self.embedding_dim,
                'batch_size_used': self.batch_size,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'chunks': embedded_chunks
            }
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ Created {successful_embeddings}/{len(chunks)} embeddings → {output_filename}")
            
            return {
                'episode_name': episode_name,
                'success': True,
                'chunks_processed': successful_embeddings,
                'output_file': output_filename,
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            return {
                'episode_name': episode_name,
                'success': False,
                'error': f'Failed to save embeddings: {e}',
                'chunks_processed': successful_embeddings
            }
    
    def process_all_chunks(self, chunks_directory: str, embeddings_output_directory: str):
        """
        Process all chunk files and create embeddings using batch processing.
        
        Args:
            chunks_directory: Directory containing chunk files
            embeddings_output_directory: Directory to save embedding files
        """
        # Create output directory
        os.makedirs(embeddings_output_directory, exist_ok=True)
        
        # Find all chunk files
        chunk_files = []
        for filename in os.listdir(chunks_directory):
            if filename.endswith('_chunks.txt'):
                chunk_files.append(os.path.join(chunks_directory, filename))
        
        if not chunk_files:
            print(f"No chunk files found in {chunks_directory}")
            return
        
        print(f"Found {len(chunk_files)} chunk files to process")
        print(f"Using embedding model: {self.model}")
        print(f"Batch size: {self.batch_size}")
        print(f"Output directory: {embeddings_output_directory}")
        print()
        
        successful_episodes = []
        failed_episodes = []
        total_chunks_processed = 0
        
        start_time = time.time()
        
        # Process each chunk file
        for i, chunk_filepath in enumerate(chunk_files, 1):
            print(f"\n[{i}/{len(chunk_files)}] ", end="")
            
            result = self.process_chunk_file(chunk_filepath, embeddings_output_directory)
            
            if result['success']:
                successful_episodes.append(result)
                total_chunks_processed += result['chunks_processed']
            else:
                failed_episodes.append(result)
                print(f"  ❌ {result['error']}")
            
            # Show progress
            elapsed = time.time() - start_time
            avg_time_per_episode = elapsed / i
            remaining_episodes = len(chunk_files) - i
            estimated_remaining = avg_time_per_episode * remaining_episodes
            
            print(f"    ⏱️  Progress: {i}/{len(chunk_files)} episodes | "
                  f"Elapsed: {elapsed/60:.1f}m | "
                  f"ETA: {estimated_remaining/60:.1f}m")
        
        total_time = time.time() - start_time
        
        # Create summary
        summary = {
            'total_episodes': len(chunk_files),
            'successful_episodes': len(successful_episodes),
            'failed_episodes': len(failed_episodes),
            'total_chunks_processed': total_chunks_processed,
            'embedding_model': self.model,
            'embedding_dimension': self.embedding_dim,
            'batch_size': self.batch_size,
            'total_processing_time_minutes': total_time / 60,
            'average_chunks_per_minute': total_chunks_processed / (total_time / 60) if total_time > 0 else 0,
            'output_directory': embeddings_output_directory,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'successful_files': successful_episodes,
            'failed_files': failed_episodes
        }
        
        summary_filepath = os.path.join(embeddings_output_directory, 'embeddings_summary.json')
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎯 EMBEDDING GENERATION SUMMARY")
        print("=" * 60)
        print(f"📁 Total episodes processed: {summary['total_episodes']}")
        print(f"✅ Successful episodes: {summary['successful_episodes']}")
        print(f"❌ Failed episodes: {summary['failed_episodes']}")
        print(f"📝 Total chunks embedded: {summary['total_chunks_processed']:,}")
        print(f"🤖 Embedding model: {summary['embedding_model']}")
        print(f"📊 Embedding dimension: {summary['embedding_dimension']}")
        print(f"📦 Batch size: {summary['batch_size']}")
        print(f"⏱️  Total time: {summary['total_processing_time_minutes']:.1f} minutes")
        print(f"🚀 Processing speed: {summary['average_chunks_per_minute']:.1f} chunks/minute")
        print(f"💾 Output directory: {embeddings_output_directory}")
        print(f"📋 Summary saved to: {summary_filepath}")
        
        if successful_episodes:
            print(f"\n📋 Example embedding files created:")
            for i, episode in enumerate(successful_episodes[:5]):
                print(f"   {i+1}. {episode['output_file']} ({episode['chunks_processed']:,} chunks)")
            if len(successful_episodes) > 5:
                print(f"   ... and {len(successful_episodes) - 5} more files")

def show_embedding_example(embeddings_directory: str):
    """
    Show an example of what an embedding file looks like.
    """
    for filename in os.listdir(embeddings_directory):
        if filename.endswith('_embeddings.json'):
            filepath = os.path.join(embeddings_directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"\n📝 Example embedding from {filename}:")
                print("-" * 50)
                print(f"Episode: {data['episode_name']}")
                print(f"Total chunks: {data['total_chunks']}")
                print(f"Embedding model: {data['embedding_model']}")
                print(f"Embedding dimension: {data['embedding_dimension']}")
                
                if data['chunks']:
                    first_chunk = data['chunks'][0]
                    print(f"\nFirst chunk:")
                    print(f"  Speaker: {first_chunk['speaker']}")
                    print(f"  Time: {first_chunk['time']}")
                    print(f"  Transcript: {first_chunk['transcript'][:100]}...")
                    print(f"  Embedding: [{first_chunk['embedding'][0]:.6f}, {first_chunk['embedding'][1]:.6f}, ..., {first_chunk['embedding'][-1]:.6f}]")
                    print(f"  Embedding length: {len(first_chunk['embedding'])}")
                
                print("-" * 50)
                break
                
            except Exception as e:
                continue

if __name__ == "__main__":
    # Configuration
    chunks_dir = "lex_fridman_chunks"        # Directory with chunk files
    embeddings_output_dir = "lex_fridman_embeddings"  # Directory for embedding files
    
    print("🤖 Creating OpenAI embeddings for Lex Fridman transcript chunks...")
    print(f"📂 Input directory: {chunks_dir}")
    print(f"📁 Output directory: {embeddings_output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(chunks_dir):
        print(f"❌ Input directory '{chunks_dir}' not found!")
        print("Make sure you've run the chunk conversion script first.")
        exit(1)
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not found!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    print(f"✅ OpenAI API key found")
    print()
    
    # Create embedding generator
    generator = ChunkEmbeddingGenerator(api_key=api_key)
    
    # Process all chunks
    generator.process_all_chunks(chunks_dir, embeddings_output_dir)
    
    # Show an example
    show_embedding_example(embeddings_output_dir) 