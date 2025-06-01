# 🎙️ Lex Fridman Transcript Search Engine

A powerful semantic search engine for Lex Fridman podcast transcripts, powered by OpenAI embeddings and LanceDB vector database.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-red)
![LanceDB](https://img.shields.io/badge/LanceDB-0.22+-orange)

## 📋 Overview

This project creates a complete pipeline for:
1. **Scraping** Lex Fridman podcast transcripts
2. **Chunking** transcripts into searchable segments
3. **Embedding** chunks using OpenAI's `text-embedding-3-small` model
4. **Storing** embeddings in LanceDB vector database
5. **Searching** with both semantic and text-based queries
6. **Web Interface** for easy searching and browsing

## 🚀 Features

### 🔍 **Semantic Search**
- AI-powered search using OpenAI embeddings
- Find conceptually similar content, not just keyword matches
- Search across 52,827 transcript segments from 98 episodes

### 🎯 **Advanced Filtering**
- Filter by specific episodes
- Filter by speaker (Lex Fridman, guests, etc.)
- Adjustable result limits (5-50 results)

### ⚡ **High Performance**
- Blazing fast vector similarity search with LanceDB
- Batch processing for efficient embedding generation
- 3,455+ chunks processed per minute during ingestion

### 🌐 **Beautiful Web Interface**
- Modern, responsive design
- Real-time search with loading states
- Mobile-optimized interface
- Dark theme with gradient accents

### 📊 **Rich Results**
- Episode information and metadata
- Speaker identification
- Timestamp information
- Similarity scores for semantic search

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API Key (for embeddings)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd memory-rft
```

2. **Initialize the project with uv:**
```bash
uv init
```

3. **Install dependencies:**
```bash
uv add requests beautifulsoup4 openai lancedb fastapi uvicorn python-multipart jinja2 tqdm
```

4. **Set up environment variables:**
```bash
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

## 🔄 Usage Pipeline

### Step 1: Scrape Transcripts
```bash
.venv/bin/python3 transcript_scraper.py
```
- Scrapes latest 100 episode transcripts from lexfridman.com
- Outputs individual JSON files in `lex_fridman_transcripts/`
- Creates summary with metadata

### Step 2: Convert to Chunks
```bash
.venv/bin/python3 transcript_to_chunks.py
```
- Converts transcripts to vector database format
- Each chunk contains: episode, speaker, transcript, time
- Outputs XML-formatted chunks in `lex_fridman_chunks/`

### Step 3: Generate Embeddings
```bash
source .env && export OPENAI_API_KEY && .venv/bin/python3 create_embeddings.py
```
- Creates OpenAI embeddings for all chunks using batch processing
- Uses `text-embedding-3-small` model (1536 dimensions)
- Outputs embedding files in `lex_fridman_embeddings/`

### Step 4: Ingest to LanceDB
```bash
.venv/bin/python3 ingest_to_lancedb.py
```
- Loads all embeddings into LanceDB vector database
- Creates optimized indexes for fast similarity search
- Database stored in `lex_fridman_vectordb/`

### Step 5: Start Web Interface
```bash
source .env && export OPENAI_API_KEY && .venv/bin/python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- Starts FastAPI web server on http://localhost:8000
- Provides both web interface and REST API

## 🌐 Web Interface

Visit `http://localhost:8000` to access the search interface.

### Features:
- **Semantic Search**: Enter natural language queries
- **Episode Filter**: Search within specific episodes
- **Speaker Filter**: Find content from specific speakers
- **Adjustable Results**: Choose 5-50 results per search
- **Real-time Stats**: Search timing and result counts
- **Responsive Design**: Works on desktop and mobile

### Example Searches:
- "artificial intelligence consciousness"
- "meaning of life philosophy"
- "future of technology"
- "quantum physics simulation"

## 🔌 API Endpoints

### Search Endpoints
- `POST /search` - Semantic search with JSON request/response
- `POST /search_form` - Form-based search for web interface

### Data Endpoints
- `GET /api/episodes` - List all available episodes
- `GET /api/speakers` - List all speakers
- `GET /api/stats` - Database statistics

### Example API Usage:
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "artificial intelligence",
       "limit": 10,
       "episode_filter": null,
       "speaker_filter": null
     }'
```

## 📊 Performance Metrics

### Embedding Generation:
- **52,827 chunks** processed in **15.3 minutes**
- **3,455 chunks/minute** using batch processing
- **98 episodes** with 100% success rate

### Database Ingestion:
- **52,827 records** inserted in **0.3 minutes**
- **177,528 records/minute** insertion rate
- **Zero failed ingestions**

### Search Performance:
- **Sub-second** semantic search response times
- **Vector similarity** using optimized LanceDB indexes
- **Real-time filtering** by episode and speaker

## 🗂️ Project Structure

```
memory-rft/
├── transcript_scraper.py          # Scrapes podcast transcripts
├── transcript_to_chunks.py        # Converts transcripts to chunks
├── create_embeddings.py           # Generates OpenAI embeddings
├── ingest_to_lancedb.py          # Loads data into LanceDB
├── api.py                        # FastAPI web application
├── test_transcript_scraper.py    # Unit tests
├── templates/
│   └── index.html               # Web interface template
├── lex_fridman_transcripts/     # Individual transcript JSON files
├── lex_fridman_chunks/          # Chunked transcript files
├── lex_fridman_embeddings/      # Embedding JSON files
├── lex_fridman_vectordb/        # LanceDB database files
├── .env                         # Environment variables
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## 🧪 Testing

Run the test suite:
```bash
.venv/bin/python3 test_transcript_scraper.py
```

Tests cover:
- Transcript URL extraction
- HTML parsing
- Data validation
- Error handling

## 🔧 Configuration

### Environment Variables:
- `OPENAI_API_KEY` - Required for embedding generation and semantic search

### Customizable Parameters:
- **Embedding Model**: Change in `create_embeddings.py` (default: `text-embedding-3-small`)
- **Batch Size**: Adjust in embedding generation (default: 100)
- **Search Limits**: Modify in web interface (5-50 results)
- **Database Path**: Configure in ingestion script

## 🚀 Deployment

### Local Development:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Production:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (optional):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Lex Fridman** for the amazing podcast content
- **OpenAI** for the embedding models
- **LanceDB** for the vector database
- **FastAPI** for the web framework

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the existing documentation
- Review the test suite for examples

---

**Built with ❤️ using Python, OpenAI, LanceDB, and FastAPI**
