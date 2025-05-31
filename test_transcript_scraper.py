import unittest
from unittest.mock import patch, MagicMock
import requests # Import requests for requests.exceptions.RequestException
from urllib.parse import urljoin

# Assuming transcript_scraper.py is in the same directory or accessible in PYTHONPATH
from transcript_scraper import get_latest_episode_transcript_urls, scrape_transcript

# Helper to create a mock response object for requests.get
def mock_response(content, status_code=200, raise_for_status_effect=None, url="http://mockurl.com"):
    mock_resp = MagicMock()
    mock_resp.content = content.encode('utf-8') # BeautifulSoup expects bytes
    mock_resp.status_code = status_code
    mock_resp.url = url # Some functions might use this
    if raise_for_status_effect:
        mock_resp.raise_for_status.side_effect = raise_for_status_effect
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp

class TestTranscriptScraper(unittest.TestCase):

    @patch('requests.get')
    def test_get_latest_episode_transcript_urls_success(self, mock_get):
        base_url = "https://lexfridman.com/podcast"
        html_content = f'''
        <html><body>
            <div><a href="/transcript-1"> [ Transcript ] </a></div>
            <div><a href="{base_url}/transcript-2"> [ Transcript ] </a></div>
            <div><a href="https://othersite.com/transcript-3"> [ Transcript ] </a></div>
            <div><a href="/no-transcript-link"> [ Video ] </a></div>
        </body></html>
        '''
        mock_get.return_value = mock_response(html_content, url=base_url)
        
        urls = get_latest_episode_transcript_urls(base_url)
        self.assertEqual(len(urls), 3)
        self.assertIn(urljoin(base_url, "/transcript-1"), urls)
        self.assertIn(f"{base_url}/transcript-2", urls) # Already absolute
        self.assertIn("https://othersite.com/transcript-3", urls)

    @patch('requests.get')
    def test_get_latest_episode_transcript_urls_limit(self, mock_get):
        base_url = "https://lexfridman.com/podcast"
        html_content = f'''
        <html><body>
            <div><a href="/t1"> [ Transcript ] </a></div>
            <div><a href="/t2"> [ Transcript ] </a></div>
            <div><a href="/t3"> [ Transcript ] </a></div>
        </body></html>
        '''
        mock_get.return_value = mock_response(html_content, url=base_url)
        
        urls = get_latest_episode_transcript_urls(base_url, limit=2)
        self.assertEqual(len(urls), 2)
        self.assertIn(urljoin(base_url, "/t1"), urls)
        self.assertIn(urljoin(base_url, "/t2"), urls)

    @patch('requests.get')
    def test_get_latest_episode_transcript_urls_no_links_found(self, mock_get):
        base_url = "https://lexfridman.com/podcast"
        html_content = "<html><body><div>No transcript links here.</div></body></html>"
        mock_get.return_value = mock_response(html_content, url=base_url)
        
        urls = get_latest_episode_transcript_urls(base_url)
        self.assertEqual(len(urls), 0)

    @patch('requests.get')
    def test_get_latest_episode_transcript_urls_request_error(self, mock_get):
        base_url = "https://lexfridman.com/podcast"
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        urls = get_latest_episode_transcript_urls(base_url)
        self.assertEqual(len(urls), 0)

    @patch('requests.get')
    def test_scrape_transcript_success_full_data(self, mock_get):
        transcript_url = "https://lexfridman.com/test-episode-transcript"
        html_content = '''
        <html><head><title>Test Page</title></head><body>
          <h1 class="entry-title"> Test Episode Title </h1>
          <div class="ts-segment">
            <span class="ts-name">Speaker One</span>
            <span class="ts-timestamp"><a href="https://youtube.com/watch?v=123&t=65">(00:01:05)</a></span>
            <span class="ts-text">First line.</span>
          </div>
          <div class="ts-segment">
            <span class="ts-name">Speaker Two</span>
            <span class="ts-timestamp">(00:02:10)</span> <!-- No href -->
            <span class="ts-text">Second line.</span>
          </div>
          <div class="ts-segment">
            <span class="ts-name"></span> <!-- Missing speaker -->
            <span class="ts-timestamp"><a href="https://youtube.com/watch?v=123&t=150">(00:02:30)</a></span>
            <span class="ts-text">Third line, default speaker.</span>
          </div>
           <div class="ts-segment">
            <span class="ts-name">Speaker One</span>
            <span class="ts-timestamp"><a href="https://youtube.com/watch?v=123&t=xyz">(00:03:00)</a></span> <!-- Bad t param -->
            <span class="ts-text">Fourth line, bad time param.</span>
          </div>
        </body></html>
        '''
        mock_get.return_value = mock_response(html_content, url=transcript_url)
        result = scrape_transcript(transcript_url)

        self.assertIsNotNone(result)
        self.assertEqual(result['episode'], "Test Episode Title")
        self.assertIn("Speaker One", result['speakers'])
        self.assertIn("Speaker Two", result['speakers'])
        self.assertIn("Lex Fridman", result['speakers']) # Default for missing
        self.assertEqual(len(result['speakers']), 3) 
        self.assertEqual(len(result['content']), 4)

        # Check first segment
        self.assertEqual(result['content'][0]['speaker'], "Speaker One")
        self.assertEqual(result['content'][0]['time'], "00:01:05") # Parsed from t=65
        self.assertEqual(result['content'][0]['transcript'], "First line.")

        # Check second segment (time from text)
        self.assertEqual(result['content'][1]['speaker'], "Speaker Two")
        self.assertEqual(result['content'][1]['time'], "00:02:10")
        self.assertEqual(result['content'][1]['transcript'], "Second line.")
        
        # Check third segment (default speaker)
        self.assertEqual(result['content'][2]['speaker'], "Lex Fridman")
        self.assertEqual(result['content'][2]['time'], "00:02:30") # Parsed from t=150
        self.assertEqual(result['content'][2]['transcript'], "Third line, default speaker.")

        # Check fourth segment (bad time param, fallback to text)
        self.assertEqual(result['content'][3]['speaker'], "Speaker One")
        self.assertEqual(result['content'][3]['time'], "00:03:00")
        self.assertEqual(result['content'][3]['transcript'], "Fourth line, bad time param.")


    @patch('requests.get')
    def test_scrape_transcript_request_error(self, mock_get):
        transcript_url = "https://lexfridman.com/network-error-transcript"
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        result = scrape_transcript(transcript_url)
        self.assertIsNone(result)

    @patch('requests.get')
    def test_scrape_transcript_http_error(self, mock_get):
        transcript_url = "https://lexfridman.com/404-transcript"
        mock_get.return_value = mock_response("Not Found", status_code=404, raise_for_status_effect=requests.exceptions.HTTPError("404 Client Error"))
        result = scrape_transcript(transcript_url)
        self.assertIsNone(result)

    @patch('requests.get')
    def test_scrape_transcript_no_title(self, mock_get):
        transcript_url = "https://lexfridman.com/no-title-transcript"
        html_content = "<html><body><div class='ts-segment'></div></body></html>" # No h1.entry-title
        mock_get.return_value = mock_response(html_content, url=transcript_url)
        result = scrape_transcript(transcript_url)
        self.assertIsNotNone(result)
        self.assertEqual(result['episode'], "Unknown Episode")

    @patch('requests.get')
    def test_scrape_transcript_no_segments_but_not_podcast_page(self, mock_get):
        transcript_url = "https://lexfridman.com/empty-transcript-page"
        # Contains no ts-segment, and not the main podcast page structure
        html_content = "<html><body><h1>Some Other Page</h1><p>No transcript here.</p></body></html>"
        mock_get.return_value = mock_response(html_content, url=transcript_url)
        result = scrape_transcript(transcript_url)
        # This will currently try the paragraph fallback, which will likely find nothing or return None
        # Based on current fallback, it will print "Could not find transcript content using fallback method either." and return None
        self.assertIsNone(result)


    @patch('requests.get')
    def test_scrape_transcript_is_podcast_listing_page(self, mock_get):
        # Simulate feeding the main podcast page URL to scrape_transcript
        podcast_listing_url = "https://lexfridman.com/podcast"
        # A simplified version of the podcast listing page structure
        html_content = "<html><body><h1>Lex Fridman Podcast</h1><a href='/t1'>[ Transcript ]</a></body></html>"
        mock_get.return_value = mock_response(html_content, url=podcast_listing_url)
        result = scrape_transcript(podcast_listing_url)
        # The function should detect this isn't a transcript page and return None
        self.assertIsNone(result)
        
    @patch('requests.get')
    def test_scrape_transcript_default_speaker_if_all_empty_and_content_exists(self, mock_get):
        transcript_url = "https://lexfridman.com/all-empty-speakers"
        html_content = '''
        <html><head><title>Test Page</title></head><body>
          <h1 class="entry-title">All Empty Speakers Episode</h1>
          <div class="ts-segment">
            <span class="ts-name"></span> <!-- Missing speaker -->
            <span class="ts-timestamp">(00:01:00)</span>
            <span class="ts-text">Line 1.</span>
          </div>
          <div class="ts-segment">
            <span class="ts-name">  </span> <!-- Whitespace speaker -->
            <span class="ts-timestamp">(00:02:00)</span>
            <span class="ts-text">Line 2.</span>
          </div>
        </body></html>
        '''
        mock_get.return_value = mock_response(html_content, url=transcript_url)
        result = scrape_transcript(transcript_url)
        self.assertIsNotNone(result)
        self.assertEqual(result['episode'], "All Empty Speakers Episode")
        # Even though individual speaker tags were empty, the logic defaults them to "Lex Fridman"
        # And the final speakers list should contain "Lex Fridman"
        self.assertEqual(result['speakers'], ["Lex Fridman"])
        self.assertEqual(len(result['content']), 2)
        self.assertEqual(result['content'][0]['speaker'], "Lex Fridman")
        self.assertEqual(result['content'][1]['speaker'], "Lex Fridman")


if __name__ == '__main__':
    unittest.main() 