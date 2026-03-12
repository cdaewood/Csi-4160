from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import httpx

app = FastAPI(title="Word Finder API", version="0.1")

Mode = Literal["word", "description"]

class WordFinderRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: Mode = "description"
    context: Optional[str] = None
    tone: Optional[str] = None
    max_results: int = Field(5, ge=1, le=10)
    use_mock: bool = False  

class WordCandidate(BaseModel):
    word: str
    part_of_speech: Optional[str] = None
    definition: Optional[str] = None
    synonyms: List[str] = []
    examples: List[str] = []
    score: float = 0.0
    why_it_fits: Optional[str] = None

class WordFinderResponse(BaseModel):
    query: str
    results: List[WordCandidate]
    sources_used: List[str]
    warnings: List[str] = []

MOCK_RESPONSE = WordFinderResponse(
    query="a word meaning extremely happy",
    results=[
        WordCandidate(
            word="elated",
            part_of_speech="adjective",
            definition="very happy or joyful",
            synonyms=["thrilled", "overjoyed", "ecstatic"],
            examples=["She felt elated after receiving the offer."],
            score=0.94,
            why_it_fits="Matches 'extremely happy' and fits formal tone."
        )
    ],
    sources_used=["mock_dictionary", "mock_datamuse"]
)

async def fetch_dictionary(word: str):
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
    if r.status_code != 200:
        return None
    return r.json()

async def fetch_datamuse_related(term: str, max_results: int):
    url = "https://api.datamuse.com/words"
    params = {"ml": term, "max": max_results}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
    if r.status_code != 200:
        return []
    return r.json()

def extract_definition_pos(dict_json):
    try:
        meaning = dict_json[0]["meanings"][0]
        pos = meaning.get("partOfSpeech")
        defs = meaning.get("definitions", [])
        definition = defs[0].get("definition") if defs else None
        example = defs[0].get("example") if defs else None
        examples = [example] if example else []
        return pos, definition, examples
    except Exception:
        return None, None, []

@app.post("/word-finder", response_model=WordFinderResponse)
async def word_finder(req: WordFinderRequest):
    if req.use_mock:
        return MOCK_RESPONSE

    # External API #1: Datamuse (semantic)
    related = await fetch_datamuse_related(req.query, req.max_results)
    if not related:
        raise HTTPException(status_code=502, detail="Semantic API failed or returned no results")

    candidates = []
    warnings = []
    sources = ["datamuse_semantic_api"]

    # Take top related words and enrich with dictionary data
    top_words = [item["word"] for item in related[:req.max_results]]

    for w in top_words:
        # External API #2: Dictionary API
        dict_json = await fetch_dictionary(w)
        if dict_json:
            sources.append("dictionaryapi_dev")
            pos, definition, examples = extract_definition_pos(dict_json)
        else:
            pos, definition, examples = None, None, []
            warnings.append(f"No dictionary data for '{w}'")

        # Simple scoring from Datamuse 'score' (scaled)
        score = 0.5
        try:
            score = float(next(item["score"] for item in related if item["word"] == w)) / 100000
        except Exception:
            pass

        candidates.append(
            WordCandidate(
                word=w,
                part_of_speech=pos,
                definition=definition,
                examples=examples,
                score=round(min(score, 1.0), 3),
                why_it_fits=f"Related to '{req.query}' via semantic API; definition provided by dictionary API."
            )
        )

    candidates.sort(key=lambda x: x.score, reverse=True)
    sources = list(dict.fromkeys(sources))

    return WordFinderResponse(
        query=req.query,
        results=candidates,
        sources_used=sources,
        warnings=warnings
    )

@app.get("/health")
def health():
    return {"status": "ok"}
