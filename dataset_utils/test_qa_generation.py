#!/usr/bin/env python3
"""
Test QA generation with a single episode to verify setup.
"""

import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def test_gemini_connection():
    """Test if Gemini API is working."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Say 'Hello, QA generation is ready!'")
        print("✓ Gemini API connection successful!")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"✗ Gemini API error: {e}")
        return False

def test_single_qa_generation():
    """Test QA generation with a small transcript excerpt."""
    # Sample transcript excerpt
    sample_transcript = """
    Lex Fridman: The following is a conversation with Oliver Anthony, a singer-songwriter from Virginia who first gained worldwide fame with his viral hit Rich Men North of Richmond.

    Oliver Anthony: Well, you were there. I'd have been doing it too, if you were out there. Like, "Oh, that's Lex Fridman."

    Lex Fridman: No, man. He was this big dude on a keyboard, just everything, sweaty, long hair, you could tell he was there in his own little world. I love the courage of that, of just giving it everything.

    Oliver Anthony: Yeah. Well, I think professionalism in general … Like, applying the tactics of corporate America to anything that is baseline artistic is not going to end well.

    Lex Fridman: They're all individually brilliant, but together, this corporate speak comes out. Just the soul of the people, it dissipates. It disappears.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""Generate 3 question-answer pairs from this podcast transcript excerpt.
        
Each question should require synthesizing information from multiple parts of the text.

Return in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "..."
    }}
  ]
}}

TRANSCRIPT:
{sample_transcript}
"""
        
        response = model.generate_content(prompt)
        result_text = response.text
        
        # Clean up response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        qa_data = json.loads(result_text.strip())
        
        print("\n✓ QA generation successful!")
        print(f"\nGenerated {len(qa_data['qa_pairs'])} QA pairs:")
        
        for i, qa in enumerate(qa_data['qa_pairs'], 1):
            print(f"\n{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer'][:150]}...")
        
        return True
        
    except Exception as e:
        print(f"\n✗ QA generation error: {e}")
        return False

def test_transcript_loading():
    """Test loading and preprocessing a real transcript."""
    transcript_dir = "dataset/lex_fridman_transcripts"
    
    # Find a transcript file
    transcript_files = [f for f in os.listdir(transcript_dir) 
                       if f.endswith('.json') and f != 'scraping_summary.json']
    
    if not transcript_files:
        print("✗ No transcript files found")
        return False
    
    # Load first transcript
    test_file = transcript_files[0]
    file_path = os.path.join(transcript_dir, test_file)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n✓ Successfully loaded: {test_file}")
        print(f"  Episode: {data['episode'][:80]}...")
        print(f"  Speakers: {', '.join(data['speakers'])}")
        print(f"  Content entries: {len(data['content'])}")
        
        # Test preprocessing
        lines = []
        for i, entry in enumerate(data["content"][:5]):  # First 5 entries
            speaker = entry["speaker"]
            text = entry["transcript"]
            lines.append(f"{speaker}: {text}")
        
        preview = "\n\n".join(lines)
        print(f"\nPreprocessed preview:\n{preview[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading transcript: {e}")
        return False

def main():
    print("QA Generation Test Suite")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("GOOGLE_API_KEY"):
        print("✗ GOOGLE_API_KEY not found in .env file")
        print("  Please add: GOOGLE_API_KEY=your_api_key")
        return
    
    print("✓ Google API key found")
    
    # Run tests
    tests = [
        ("Gemini API Connection", test_gemini_connection),
        ("Single QA Generation", test_single_qa_generation),
        ("Transcript Loading", test_transcript_loading)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n\nTesting: {test_name}")
        print("-" * 30)
        if not test_func():
            all_passed = False
    
    print("\n\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! Ready to generate QA pairs.")
        print("\nRun full generation with:")
        print("  python dataset_utils/generate_qa_pairs.py --max-episodes 1")
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")

if __name__ == "__main__":
    main() 