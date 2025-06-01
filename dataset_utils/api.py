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
import random
import numpy as np
import random
import numpy as np

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

class QAPair(BaseModel):
    question: str
    answer: str
    episode: str
    episode_file: str
    difficulty: str
    requires_synthesis: bool
    qa_index: int

class QABatchRequest(BaseModel):
    batch_size: int = 32
    shuffle: bool = True
    difficulty_filter: Optional[str] = None
    episode_filter: Optional[str] = None
    seed: Optional[int] = None

class QABatchResponse(BaseModel):
    qa_pairs: List[QAPair]
    batch_size: int
    total_available: int
    seed_used: Optional[int]

class LexFridmanSearchAPI:
    def __init__(self, db_path: str = "dataset/lex_fridman_vectordb", table_name: str = "lex_fridman_transcripts", qa_table_name: str = "lex_fridman_qa"):
        """
        Initialize the search API.
        
        Args:
            db_path: Path to the LanceDB database
            table_name: Name of the table containing transcripts
            qa_table_name: Name of the table containing QA pairs
        """
        self.db_path = db_path
        self.table_name = table_name
        self.qa_table_name = qa_table_name
        self.db = None
        self.table = None
        self.qa_table = None
        self.openai_client = None
        self.qa_pairs_cache = None
        self.qa_cache_time = None
        self.cache_duration = 300  # Cache QA pairs for 5 minutes
        self.connect_db()
        
    def connect_db(self):
        """Connect to the LanceDB database."""
        try:
            self.db = lancedb.connect(self.db_path)
            self.table = self.db.open_table(self.table_name)
            print(f"✅ Connected to LanceDB at {self.db_path}")
            
            # Try to connect to QA table
            try:
                self.qa_table = self.db.open_table(self.qa_table_name)
                print(f"✅ Connected to QA table '{self.qa_table_name}'")
            except Exception as e:
                print(f"⚠️  QA table '{self.qa_table_name}' not found - QA endpoints will be unavailable")
                self.qa_table = None
            
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
    
    def get_qa_pairs(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all QA pairs, using cache if available.
        
        Args:
            force_refresh: Force refresh the cache
            
        Returns:
            List of QA pairs
        """
        if not self.qa_table:
            return []
        
        # Check if cache is valid
        if (not force_refresh and 
            self.qa_pairs_cache is not None and 
            self.qa_cache_time is not None and
            time.time() - self.qa_cache_time < self.cache_duration):
            return self.qa_pairs_cache
        
        try:
            # Get all QA questions (we'll use questions, not answers)
            results = (self.qa_table
                      .search()
                      .where("type = 'qa_question'")
                      .to_list())
            
            self.qa_pairs_cache = results
            self.qa_cache_time = time.time()
            print(f"✅ Loaded {len(results)} QA pairs into cache")
            
            return results
        except Exception as e:
            print(f"Error loading QA pairs: {e}")
            return []
    
    def get_qa_batch(self, batch_size: int = 32, shuffle: bool = True, 
                     difficulty_filter: Optional[str] = None,
                     episode_filter: Optional[str] = None,
                     seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a batch of QA pairs with optional shuffling and filtering.
        
        Args:
            batch_size: Number of QA pairs to return
            shuffle: Whether to shuffle the pairs
            difficulty_filter: Filter by difficulty level
            episode_filter: Filter by episode
            seed: Random seed for reproducible shuffling
            
        Returns:
            Dictionary containing the batch and metadata
        """
        # Get all QA pairs
        all_qa_pairs = self.get_qa_pairs()
        
        if not all_qa_pairs:
            return {
                "qa_pairs": [],
                "batch_size": 0,
                "total_available": 0,
                "seed_used": None
            }
        
        # Apply filters
        filtered_pairs = all_qa_pairs
        
        if difficulty_filter:
            filtered_pairs = [qa for qa in filtered_pairs 
                            if qa.get('difficulty', '').lower() == difficulty_filter.lower()]
        
        if episode_filter:
            filtered_pairs = [qa for qa in filtered_pairs 
                            if qa.get('episode_file', '').lower() == episode_filter.lower()]
        
        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            else:
                seed = random.randint(0, 1000000)
                random.seed(seed)
                np.random.seed(seed)
            
            shuffled_pairs = filtered_pairs.copy()
            random.shuffle(shuffled_pairs)
        else:
            shuffled_pairs = filtered_pairs
            seed = None
        
        # Get batch
        batch = shuffled_pairs[:batch_size]
        
        # Format the batch
        formatted_batch = []
        for qa in batch:
            formatted_batch.append({
                "question": qa.get('text', ''),  # In DB, question text is in 'text' field
                "answer": qa.get('answer', ''),
                "episode": qa.get('episode', ''),
                "episode_file": qa.get('episode_file', ''),
                "difficulty": qa.get('difficulty', 'hard'),
                "requires_synthesis": qa.get('requires_synthesis', True),
                "qa_index": qa.get('qa_index', 0)
            })
        
        return {
            "qa_pairs": formatted_batch,
            "batch_size": len(formatted_batch),
            "total_available": len(filtered_pairs),
            "seed_used": seed
        }
    
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
    
    def get_qa_episodes(self) -> List[str]:
        """Get list of all unique episodes with QA pairs."""
        if not self.qa_table:
            return []
        
        try:
            results = (self.qa_table
                      .search()
                      .where("type = 'qa_question'")
                      .select(["episode_file"])
                      .to_list())
            
            episodes = list(set([r['episode_file'] for r in results if r.get('episode_file')]))
            return sorted(episodes)
        except Exception as e:
            print(f"Error getting QA episodes: {e}")
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
templates = Jinja2Templates(directory="dataset_utils/templates")

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

@app.get("/qa", response_class=HTMLResponse)
async def qa_interface(request: Request):
    """QA pairs interface."""
    if not search_api.qa_table:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "QA pairs not available"
        })
    
    return templates.TemplateResponse("qa.html", {
        "request": request
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

@app.post("/api/qa/batch", response_model=QABatchResponse)
async def get_qa_batch(request: QABatchRequest):
    """
    Get a shuffled batch of QA pairs for training or evaluation.
    
    Example:
    ```
    POST /api/qa/batch
    {
        "batch_size": 32,
        "shuffle": true,
        "difficulty_filter": "hard",
        "seed": 42
    }
    ```
    """
    if not search_api.qa_table:
        raise HTTPException(status_code=503, detail="QA table not available")
    
    result = search_api.get_qa_batch(
        batch_size=request.batch_size,
        shuffle=request.shuffle,
        difficulty_filter=request.difficulty_filter,
        episode_filter=request.episode_filter,
        seed=request.seed
    )
    
    return QABatchResponse(
        qa_pairs=[QAPair(**qa) for qa in result['qa_pairs']],
        batch_size=result['batch_size'],
        total_available=result['total_available'],
        seed_used=result['seed_used']
    )

@app.get("/api/qa/episodes")
async def get_qa_episodes():
    """Get list of all episodes that have QA pairs."""
    if not search_api.qa_table:
        raise HTTPException(status_code=503, detail="QA table not available")
    
    return {"episodes": search_api.get_qa_episodes()}

@app.get("/api/qa/stats")
async def get_qa_stats():
    """Get statistics about QA pairs."""
    if not search_api.qa_table:
        raise HTTPException(status_code=503, detail="QA table not available")
    
    try:
        # Get total QA pairs
        total_questions = search_api.qa_table.search().where("type = 'qa_question'").to_list()
        
        # Get difficulty distribution
        difficulty_counts = {}
        for qa in total_questions:
            difficulty = qa.get('difficulty', 'unknown')
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        # Get episode distribution
        episode_counts = {}
        for qa in total_questions:
            episode = qa.get('episode_file', 'unknown')
            episode_counts[episode] = episode_counts.get(episode, 0) + 1
        
        return {
            "total_qa_pairs": len(total_questions),
            "difficulty_distribution": difficulty_counts,
            "episodes_with_qa": len(episode_counts),
            "average_qa_per_episode": len(total_questions) / len(episode_counts) if episode_counts else 0,
            "qa_table_name": search_api.qa_table_name
        }
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
        
        # QA stats if available
        qa_stats = {}
        if search_api.qa_table:
            try:
                qa_count = len(search_api.qa_table.search().where("type = 'qa_question'").to_list())
                qa_stats = {
                    "qa_pairs_available": True,
                    "total_qa_pairs": qa_count
                }
            except:
                qa_stats = {"qa_pairs_available": False}
        else:
            qa_stats = {"qa_pairs_available": False}
        
        return {
            "total_transcripts": total_rows,
            "total_episodes": len(episodes),
            "total_speakers": len(speakers),
            "database_path": search_api.db_path,
            "semantic_search_available": search_api.openai_client is not None,
            **qa_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 