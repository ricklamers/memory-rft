import json
import os
import re
from pathlib import Path

def clean_text(text):
    """
    Clean transcript text for better vector database insertion.
    Remove extra whitespace and normalize quotes.
    """
    # Replace curly quotes with straight quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    return text.strip()

def create_chunk(episode_title, speaker, transcript, time):
    """
    Create a single chunk in the specified XML format.
    """
    # Clean the inputs
    episode_clean = clean_text(episode_title)
    speaker_clean = clean_text(speaker)
    transcript_clean = clean_text(transcript)
    time_clean = time.strip()
    
    chunk = f"""<episode>{episode_clean}</episode>
<speaker>{speaker_clean}</speaker>
<transcript>{transcript_clean}</transcript>
<time>{time_clean}</time>"""
    
    return chunk

def process_transcript_file(json_filepath, output_directory):
    """
    Process a single transcript JSON file and create chunks.
    
    Args:
        json_filepath: Path to the JSON transcript file
        output_directory: Directory to save the chunks file
    
    Returns:
        Tuple of (success, chunks_count, output_filepath)
    """
    try:
        # Read the JSON file
        with open(json_filepath, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        episode_title = transcript_data.get('episode', 'Unknown Episode')
        content = transcript_data.get('content', [])
        
        if not content:
            print(f"  ⚠️  No content found in {json_filepath}")
            return False, 0, None
        
        # Create chunks for each content segment
        chunks = []
        for segment in content:
            speaker = segment.get('speaker', 'Unknown Speaker')
            transcript = segment.get('transcript', '')
            time = segment.get('time', '00:00:00')
            
            # Skip empty transcripts
            if not transcript.strip():
                continue
            
            chunk = create_chunk(episode_title, speaker, transcript, time)
            chunks.append(chunk)
        
        if not chunks:
            print(f"  ⚠️  No valid chunks created from {json_filepath}")
            return False, 0, None
        
        # Create output filename based on input filename
        input_filename = Path(json_filepath).stem
        output_filename = f"{input_filename}_chunks.txt"
        output_filepath = os.path.join(output_directory, output_filename)
        
        # Write chunks to file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(chunk)
                if i < len(chunks) - 1:  # Don't add separator after last chunk
                    f.write('\n\n---\n\n')
        
        return True, len(chunks), output_filepath
        
    except Exception as e:
        print(f"  ❌ Error processing {json_filepath}: {e}")
        return False, 0, None

def convert_all_transcripts(transcripts_directory, chunks_output_directory):
    """
    Convert all transcript JSON files to chunks.
    
    Args:
        transcripts_directory: Directory containing JSON transcript files
        chunks_output_directory: Directory to save chunk files
    """
    # Create output directory if it doesn't exist
    os.makedirs(chunks_output_directory, exist_ok=True)
    
    # Find all JSON files (excluding the summary file)
    json_files = []
    for filename in os.listdir(transcripts_directory):
        if filename.endswith('.json') and filename != 'scraping_summary.json':
            json_files.append(os.path.join(transcripts_directory, filename))
    
    if not json_files:
        print(f"No transcript JSON files found in {transcripts_directory}")
        return
    
    print(f"Found {len(json_files)} transcript files to process")
    print(f"Output directory: {chunks_output_directory}")
    print()
    
    successful_conversions = []
    failed_conversions = []
    total_chunks = 0
    
    for i, json_filepath in enumerate(json_files, 1):
        filename = os.path.basename(json_filepath)
        print(f"Processing {i}/{len(json_files)}: {filename}")
        
        success, chunk_count, output_filepath = process_transcript_file(
            json_filepath, chunks_output_directory
        )
        
        if success:
            successful_conversions.append({
                'input_file': filename,
                'output_file': os.path.basename(output_filepath),
                'chunk_count': chunk_count
            })
            total_chunks += chunk_count
            print(f"  ✅ Created {chunk_count} chunks → {os.path.basename(output_filepath)}")
        else:
            failed_conversions.append({
                'input_file': filename,
                'error': 'Failed to process'
            })
        
        print()
    
    # Create summary file
    summary = {
        'total_files_processed': len(json_files),
        'successful_conversions': len(successful_conversions),
        'failed_conversions': len(failed_conversions),
        'total_chunks_created': total_chunks,
        'output_directory': chunks_output_directory,
        'successful_files': successful_conversions,
        'failed_files': failed_conversions
    }
    
    summary_filepath = os.path.join(chunks_output_directory, 'conversion_summary.json')
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print final summary
    print("=" * 60)
    print("🎯 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"📁 Total files processed: {summary['total_files_processed']}")
    print(f"✅ Successful conversions: {summary['successful_conversions']}")
    print(f"❌ Failed conversions: {summary['failed_conversions']}")
    print(f"📝 Total chunks created: {summary['total_chunks_created']:,}")
    print(f"💾 Output directory: {chunks_output_directory}")
    print(f"📊 Summary saved to: {summary_filepath}")
    
    if successful_conversions:
        print(f"\n📋 Example chunk files created:")
        for i, conv in enumerate(successful_conversions[:5]):
            print(f"   {i+1}. {conv['output_file']} ({conv['chunk_count']:,} chunks)")
        if len(successful_conversions) > 5:
            print(f"   ... and {len(successful_conversions) - 5} more files")

def show_chunk_example(chunks_output_directory):
    """
    Show an example of what a chunk looks like.
    """
    # Find the first chunk file
    for filename in os.listdir(chunks_output_directory):
        if filename.endswith('_chunks.txt'):
            filepath = os.path.join(chunks_output_directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Get the first chunk (before first separator)
                    first_chunk = content.split('\n\n---\n\n')[0]
                    
                    print(f"\n📝 Example chunk from {filename}:")
                    print("-" * 50)
                    print(first_chunk)
                    print("-" * 50)
                    break
            except Exception as e:
                continue

if __name__ == "__main__":
    # Configuration
    transcripts_dir = "lex_fridman_transcripts"  # Directory with JSON transcript files
    chunks_output_dir = "lex_fridman_chunks"    # Directory for chunk files
    
    print("🔄 Converting Lex Fridman transcripts to vector database chunks...")
    print(f"📂 Input directory: {transcripts_dir}")
    print(f"📁 Output directory: {chunks_output_dir}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(transcripts_dir):
        print(f"❌ Input directory '{transcripts_dir}' not found!")
        print("Make sure you've run the transcript scraper first.")
        exit(1)
    
    # Convert all transcripts
    convert_all_transcripts(transcripts_dir, chunks_output_dir)
    
    # Show an example chunk
    show_chunk_example(chunks_output_dir) 