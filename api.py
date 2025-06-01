from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import lancedb
import os
from openai import OpenAI
from pydantic import BaseModel
import json
import time

# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    episode_filter: Optional[str] = None
    speaker_filter: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    episode: str
    speaker: str
    transcript: str
    time: str
    similarity_score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float

class LexFridmanSearchAPI:
    def __init__(self, db_path: str = "lex_fridman_vectordb", table_name: str = "lex_fridman_transcripts"):
        """
        Initialize the search API.
        
        Args:
            db_path: Path to the LanceDB database
            table_name: Name of the table containing transcripts
        """
        self.db_path = db_path
        self.table_name = table_name
        self.db = None
        self.table = None
        self.openai_client = None
        self.connect_db()
        
    def connect_db(self):
        """Connect to the LanceDB database."""
        try:
            self.db = lancedb.connect(self.db_path)
            self.table = self.db.open_table(self.table_name)
            print(f"✅ Connected to LanceDB at {self.db_path}")
            
            # Initialize OpenAI client for embedding queries
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                print(f"✅ OpenAI client initialized for semantic search")
            else:
                print(f"⚠️  OpenAI API key not found - only text search available")
                
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            raise e
    
    def embed_query(self, query: str) -> List[float]:
        """
        Create an embedding for the search query.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector for the query
        """
        if not self.openai_client:
            raise HTTPException(status_code=500, detail="OpenAI client not available for semantic search")
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating embedding: {e}")
    
    def semantic_search(self, query: str, limit: int = 10, episode_filter: Optional[str] = None, 
                       speaker_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query
            limit: Number of results to return
            episode_filter: Optional episode name filter
            speaker_filter: Optional speaker name filter
            
        Returns:
            List of search results
        """
        # Get query embedding
        query_vector = self.embed_query(query)
        
        # Build search query
        search_query = self.table.search(query_vector).limit(limit)
        
        # Add filters if specified
        filters = []
        if episode_filter:
            filters.append(f"episode = '{episode_filter}'")
        if speaker_filter:
            filters.append(f"speaker = '{speaker_filter}'")
        
        if filters:
            search_query = search_query.where(" AND ".join(filters))
        
        # Execute search
        results = search_query.to_list()
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result['id'],
                'episode': result['episode'],
                'speaker': result['speaker'],
                'transcript': result['transcript'],
                'time': result['time'],
                'similarity_score': result['_distance']  # LanceDB returns distance, lower is better
            })
        
        return formatted_results
    
    def text_search(self, query: str, limit: int = 10, episode_filter: Optional[str] = None, 
                   speaker_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform text-based search using SQL LIKE.
        
        Args:
            query: Search query
            limit: Number of results to return
            episode_filter: Optional episode name filter
            speaker_filter: Optional speaker name filter
            
        Returns:
            List of search results
        """
        # Build search query
        filters = [f"transcript LIKE '%{query}%'"]
        
        if episode_filter:
            filters.append(f"episode = '{episode_filter}'")
        if speaker_filter:
            filters.append(f"speaker = '{speaker_filter}'")
        
        where_clause = " AND ".join(filters)
        
        # Execute search
        results = (self.table
                  .search()
                  .where(where_clause)
                  .limit(limit)
                  .to_list())
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result['id'],
                'episode': result['episode'],
                'speaker': result['speaker'],
                'transcript': result['transcript'],
                'time': result['time'],
                'similarity_score': 1.0  # No similarity score for text search
            })
        
        return formatted_results
    
    def get_episodes(self) -> List[str]:
        """Get list of all unique episodes."""
        try:
            # Get unique episodes from the database
            results = (self.table
                      .search()
                      .select(["episode"])
                      .to_list())
            
            episodes = list(set([r['episode'] for r in results]))
            return sorted(episodes)
        except Exception as e:
            print(f"Error getting episodes: {e}")
            return []
    
    def get_speakers(self) -> List[str]:
        """Get list of all unique speakers."""
        try:
            # Get unique speakers from the database
            results = (self.table
                      .search()
                      .select(["speaker"])
                      .to_list())
            
            speakers = list(set([r['speaker'] for r in results]))
            return sorted(speakers)
        except Exception as e:
            print(f"Error getting speakers: {e}")
            return []

# Initialize the API
app = FastAPI(title="Lex Fridman Transcript Search", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Initialize search engine
search_api = LexFridmanSearchAPI()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search interface."""
    episodes = search_api.get_episodes()
    speakers = search_api.get_speakers()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "episodes": episodes,
        "speakers": speakers
    })

@app.post("/search", response_model=SearchResponse)
async def search_transcripts(request: SearchRequest):
    """
    Search transcripts using semantic or text search.
    """
    start_time = time.time()
    
    try:
        # Try semantic search first if OpenAI is available
        if search_api.openai_client:
            results = search_api.semantic_search(
                query=request.query,
                limit=request.limit,
                episode_filter=request.episode_filter,
                speaker_filter=request.speaker_filter
            )
        else:
            # Fall back to text search
            results = search_api.text_search(
                query=request.query,
                limit=request.limit,
                episode_filter=request.episode_filter,
                speaker_filter=request.speaker_filter
            )
        
        search_time = (time.time() - start_time) * 1000
        
        # Convert to response format
        search_results = [
            SearchResult(
                id=r['id'],
                episode=r['episode'],
                speaker=r['speaker'],
                transcript=r['transcript'],
                time=r['time'],
                similarity_score=r['similarity_score']
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_form", response_class=HTMLResponse)
async def search_form(
    request: Request,
    query: str = Form(...),
    limit: int = Form(10),
    episode_filter: str = Form(""),
    speaker_filter: str = Form("")
):
    """
    Handle form-based search for the web interface.
    """
    # Convert empty strings to None
    episode_filter = episode_filter if episode_filter else None
    speaker_filter = speaker_filter if speaker_filter else None
    
    search_request = SearchRequest(
        query=query,
        limit=limit,
        episode_filter=episode_filter,
        speaker_filter=speaker_filter
    )
    
    search_response = await search_transcripts(search_request)
    
    episodes = search_api.get_episodes()
    speakers = search_api.get_speakers()
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": search_response.results,
        "query": query,
        "search_time": search_response.search_time_ms,
        "total_results": search_response.total_results,
        "episodes": episodes,
        "speakers": speakers,
        "selected_episode": episode_filter,
        "selected_speaker": speaker_filter
    })

@app.get("/api/episodes")
async def get_episodes():
    """Get list of all episodes."""
    return {"episodes": search_api.get_episodes()}

@app.get("/api/speakers") 
async def get_speakers():
    """Get list of all speakers."""
    return {"speakers": search_api.get_speakers()}

@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    try:
        total_rows = search_api.table.count_rows()
        episodes = search_api.get_episodes()
        speakers = search_api.get_speakers()
        
        return {
            "total_transcripts": total_rows,
            "total_episodes": len(episodes),
            "total_speakers": len(speakers),
            "database_path": search_api.db_path,
            "semantic_search_available": search_api.openai_client is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 