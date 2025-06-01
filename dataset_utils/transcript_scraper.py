import requests
from bs4 import BeautifulSoup
import json
import os
import re
from urllib.parse import urljoin, urlparse # Added for joining relative URLs

def get_latest_episode_transcript_urls(podcast_page_url, limit=100):
    """
    Fetches the URLs of the latest episode transcripts from the Lex Fridman podcast page.

    Args:
        podcast_page_url: The URL of the main podcast page.
        limit: The maximum number of transcript URLs to return.

    Returns:
        A list of absolute URLs for the latest episode transcripts, up to the specified limit.
        Returns an empty list if fetching or parsing fails.
    """
    print(f"Fetching latest episode list from: {podcast_page_url}")
    try:
        response = requests.get(podcast_page_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching podcast page {podcast_page_url}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    transcript_urls = []
    
    # Find all links. We'll specifically look for ones with the text "\[ Transcript \]" (with escaped brackets)
    # The structure seems to be:
    # ... <h3>Guest Name</h3> ... <a>\[ Video \]</a> <a>\[ Episode \]</a> <a>\[ Transcript \]</a>
    # We are interested in the href of the 'a' tag that contains the text "\[ Transcript \]"
    
    # Let's find all 'a' tags and filter them - looking for both possible formats
    transcript_link_patterns = [
        "[ Transcript ]",   # Original pattern we were looking for
        "\[ Transcript \]", # Escaped brackets pattern from actual site
        "\\[ Transcript \\]" # Double-escaped in case needed
    ]
    
    for pattern in transcript_link_patterns:
        links_found = soup.find_all('a', string=lambda text: text and pattern in text.strip())
        if links_found:
            print(f"Found transcript links using pattern: '{pattern}'")
            break
    else:
        # If no pattern worked, try a more flexible approach
        links_found = soup.find_all('a', string=lambda text: text and 'Transcript' in text.strip())
        if links_found:
            print("Found transcript links using flexible 'Transcript' pattern")
        else:
            print("No transcript links found with any pattern")

    for link_tag in links_found:
        href = link_tag.get('href')
        if href:
            # Ensure the URL is absolute
            absolute_url = urljoin(podcast_page_url, href)
            transcript_urls.append(absolute_url)
            if len(transcript_urls) >= limit:
                break
    
    if not transcript_urls:
        print(f"No transcript links found on {podcast_page_url} with the expected format.")
        # Debug: Let's see what links we do have
        all_links = soup.find_all('a')
        print(f"Total links found on page: {len(all_links)}")
        transcript_like_links = [link for link in all_links if link.get_text() and 'transcript' in link.get_text().lower()]
        print(f"Links containing 'transcript': {len(transcript_like_links)}")
        if transcript_like_links:
            for i, link in enumerate(transcript_like_links[:5]):  # Show first 5
                print(f"  {i+1}: '{link.get_text().strip()}' -> {link.get('href')}")
    else:
        print(f"Found {len(transcript_urls)} transcript URLs.")
        
    return transcript_urls

def scrape_transcript(url):
    """
    Scrapes the transcript from a Lex Fridman podcast transcript page.

    Args:
        url: The URL of the transcript page.

    Returns:
        A dictionary containing the scraped transcript data in the specified JSON structure.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    episode_title_tag = soup.find('h1', class_='entry-title')
    episode_title = episode_title_tag.text.strip() if episode_title_tag else "Unknown Episode"

    # Extract speakers - we'll build this list as we parse segments
    speakers = set()
    content = []

    transcript_segments = soup.find_all('div', class_='ts-segment')

    if not transcript_segments:
        print(f"No transcript segments found on {url}. The page structure might have changed.")
        # Fallback: Try to find transcript-like paragraphs if the primary structure isn't found
        # This is a guess and might need refinement based on actual alternative structures
        
        # Check if we are trying to parse the main podcast page instead of a transcript page
        if soup.find('h1', string=lambda text: text and "Lex Fridman Podcast" in text):
             print(f"Warning: {url} appears to be a podcast listing page, not a transcript page. Skipping.")
             return None

        paragraphs = soup.find_all('p')
        # A simple heuristic: if a paragraph contains a timestamp-like pattern and a colon (speaker indicator)
        # This is highly speculative and likely needs adjustment.
        import re
        time_pattern = re.compile(r'\(\d{2}:\d{2}:\d{2}\)')
        potential_segments = []
        current_speaker = "Unknown"
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            time_match = time_pattern.search(text)
            if ":" in text and time_match: # Basic check for speaker: text (timestamp)
                parts = text.split(":", 1)
                speaker_candidate = parts[0].strip()
                transcript_text = parts[1].strip()
                # Remove timestamp from transcript text if present
                transcript_text = time_pattern.sub('', transcript_text).strip()
                
                # Attempt to extract speaker if it's different from the previous one
                # This is very naive.
                if speaker_candidate and len(speaker_candidate) < 30: # Assume speaker names are short
                    current_speaker = speaker_candidate
                
                speakers.add(current_speaker)
                content.append({
                    "speaker": current_speaker,
                    "transcript": transcript_text,
                    "time": time_match.group(1) if time_match else "00:00:00"
                })
        if not content:
             print("Could not find transcript content using fallback method either.")
             return None


    for segment in transcript_segments:
        speaker_tag = segment.find('span', class_='ts-name')
        speaker = speaker_tag.text.strip() if speaker_tag and speaker_tag.text.strip() else "Lex Fridman" # Default to Lex if no speaker tag or empty
        
        timestamp_tag = segment.find('span', class_='ts-timestamp')
        # Extract time from the format (HH:MM:SS)
        time_text = "00:00:00"
        if timestamp_tag and timestamp_tag.a:
            href = timestamp_tag.a.get('href', '')
            if 't=' in href:
                try:
                    seconds = int(href.split('t=')[-1])
                    # Convert seconds to HH:MM:SS
                    hours = seconds // 3600
                    minutes = (seconds % 3600) // 60
                    secs = seconds % 60
                    time_text = f"{hours:02d}:{minutes:02d}:{secs:02d}"
                except ValueError:
                    # Handle cases where 't=' is not a number, or format is unexpected
                    time_text = timestamp_tag.text.strip().replace('(', '').replace(')', '') if timestamp_tag.text else "00:00:00"
            else: # Fallback if t= parameter is not in href
                 time_text = timestamp_tag.text.strip().replace('(', '').replace(')', '') if timestamp_tag.text else "00:00:00"
        elif timestamp_tag: # Fallback if 'a' tag is missing
            time_text = timestamp_tag.text.strip().replace('(', '').replace(')', '') if timestamp_tag.text else "00:00:00"


        transcript_tag = segment.find('span', class_='ts-text')
        transcript = transcript_tag.text.strip() if transcript_tag else ""

        if speaker: # Add speaker to our set of unique speakers
            speakers.add(speaker)

        content.append({
            "speaker": speaker,
            "transcript": transcript,
            "time": time_text
        })
        
    # If after processing all segments, speakers set is empty but content exists,
    # it implies a single speaker (likely Lex) for all segments or an issue with speaker parsing.
    # As a fallback, if speakers set is empty and content is not, assume "Lex Fridman" as the speaker for all.
    if not speakers and content:
        speakers.add("Lex Fridman") # Default speaker
        # Update content with default speaker if necessary, though already handled by defaulting speaker in loop.

    return {
        "speakers": list(speakers) if speakers else ["Lex Fridman"], # Ensure speakers list is not empty
        "episode": episode_title,
        "content": content
    }

def scrape_multiple_episodes(episode_urls, output_directory="transcripts"):
    """
    Scrapes transcripts for multiple Lex Fridman podcast episodes and saves each as a separate file.

    Args:
        episode_urls: A list of URLs for the transcript pages.
        output_directory: Directory where individual transcript files will be saved.

    Returns:
        A list of dictionaries containing metadata about the scraped files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    successfully_scraped = []
    failed_scrapes = []
    
    for i, url in enumerate(episode_urls, 1):
        print(f"Scraping {i}/{len(episode_urls)}: {url}")
        transcript_data = scrape_transcript(url)
        
        if transcript_data:
            # Create a safe filename from the URL or episode title
            # Extract the episode identifier from URL (e.g., "oliver-anthony-transcript" from the URL)
            parsed_url = urlparse(url)
            url_path = parsed_url.path.strip('/')
            
            # Remove '-transcript' suffix if present and clean the filename
            filename_base = url_path.replace('-transcript', '').replace('/', '_')
            filename_base = re.sub(r'[^\w\-_]', '', filename_base)  # Remove special characters
            
            # Ensure filename isn't too long
            if len(filename_base) > 100:
                filename_base = filename_base[:100]
            
            filename = f"{filename_base}.json"
            filepath = os.path.join(output_directory, filename)
            
            # Handle duplicate filenames by adding a number
            counter = 1
            original_filepath = filepath
            while os.path.exists(filepath):
                name, ext = os.path.splitext(original_filepath)
                filepath = f"{name}_{counter}{ext}"
                counter += 1
            
            # Save the individual transcript file
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, ensure_ascii=False, indent=2)
                
                successfully_scraped.append({
                    "url": url,
                    "filename": os.path.basename(filepath),
                    "filepath": filepath,
                    "episode": transcript_data.get("episode", "Unknown Episode"),
                    "speakers": transcript_data.get("speakers", []),
                    "content_segments": len(transcript_data.get("content", []))
                })
                print(f"  ✓ Saved to: {filepath}")
                
            except Exception as e:
                print(f"  ✗ Error saving file: {e}")
                failed_scrapes.append({"url": url, "error": str(e)})
        else:
            print(f"  ✗ Failed to scrape transcript")
            failed_scrapes.append({"url": url, "error": "Failed to scrape transcript"})
        
        print("-" * 50)
    
    # Save a summary/index file
    summary = {
        "total_episodes": len(episode_urls),
        "successfully_scraped": len(successfully_scraped),
        "failed_scrapes": len(failed_scrapes),
        "output_directory": output_directory,
        "scraped_files": successfully_scraped,
        "failed_urls": failed_scrapes
    }
    
    summary_filepath = os.path.join(output_directory, "scraping_summary.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 SCRAPING SUMMARY:")
    print(f"   Total episodes attempted: {summary['total_episodes']}")
    print(f"   Successfully scraped: {summary['successfully_scraped']}")
    print(f"   Failed: {summary['failed_scrapes']}")
    print(f"   Output directory: {output_directory}")
    print(f"   Summary saved to: {summary_filepath}")
    
    return successfully_scraped

if __name__ == "__main__":
    podcast_home_url = "https://lexfridman.com/podcast"
    number_of_episodes_to_scrape = 100 # Target 100 latest episodes
    output_dir = "lex_fridman_transcripts"  # Directory for all transcript files

    print(f"Attempting to fetch the latest {number_of_episodes_to_scrape} transcript URLs from {podcast_home_url}")
    episode_urls_to_scrape = get_latest_episode_transcript_urls(podcast_home_url, limit=number_of_episodes_to_scrape)

    if not episode_urls_to_scrape:
        print("No episode URLs found to scrape. Exiting.")
    else:
        print(f"Proceeding to scrape {len(episode_urls_to_scrape)} transcripts into separate files.")
        print(f"Output directory: {output_dir}")
        
        scraped_files = scrape_multiple_episodes(episode_urls_to_scrape, output_directory=output_dir)

        if scraped_files:
            print(f"\n🎉 Successfully created {len(scraped_files)} transcript files!")
            
            # Show a few examples
            print("\n📝 Example transcript files created:")
            for i, file_info in enumerate(scraped_files[:5]):  # Show first 5
                print(f"   {i+1}. {file_info['filename']}")
                print(f"      Episode: {file_info['episode'][:80]}{'...' if len(file_info['episode']) > 80 else ''}")
                print(f"      Speakers: {', '.join(file_info['speakers'])}")
                print(f"      Segments: {file_info['content_segments']}")
                print()
            
            if len(scraped_files) > 5:
                print(f"   ... and {len(scraped_files) - 5} more files")
        else:
            print("❌ No transcripts were scraped successfully.") 