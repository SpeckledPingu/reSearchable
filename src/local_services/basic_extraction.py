from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import json
from unidecode import unidecode
import re
from txtai.pipeline import Summary
from fastapi import FastAPI
from pydantic import BaseModel

class Doc(BaseModel):
    doc: str

app = FastAPI()

kw_vectorizer = KeyphraseCountVectorizer()
kw_extractor = KeyBERT()
summarizer = Summary()

@app.post("/ask/tag")
def extract_keyphrases(doc: Doc):
    print(doc)
    doc = unidecode(doc.doc)
    doc = re.sub(r'(\s+)',' ', doc)

    keyphrases = kw_extractor.extract_keywords(docs=[doc], top_n=10, use_mmr=True, diversity=0.3, vectorizer=kw_vectorizer)
    keyphrases = [x[0] for x in keyphrases]
    return {'keyphrases':keyphrases}

@app.post("/ask/summary")
def extract_summary(doc: Doc):
    doc = unidecode(doc.doc)
    doc = re.sub(r'(\s+)',' ', doc)
    return {'summary':summarizer(doc)}

