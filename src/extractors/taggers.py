import spacy
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import re
from unidecode import unidecode
class SpacyExtractor():
    def __init__(self, spacy_model='en_core_web_sm', spacy_exclude=list()):
        self.nlp = spacy.load(spacy_model) \
                    if len(spacy_exclude) == 0 \
                    else spacy.load(spacy_model, exclude=spacy_exclude)

    def create_doc(self, doc):
        self.doc = self.nlp(doc)
        self.entities = list()
        self.people = list()
        self.places = list()
        self.orgs = list()
    def extract_entities(self):
        pass

    def extract_people(self):
        for ent in self.doc.ents:
            if (ent.label_ == 'PERSON') or (ent.label_ == 'NORP'):
                self.people.append(ent.text)
        if len(self.people) > 0:
            return ', '.join(list(set(self.people)))
        else:
            return ''

    def extract_places(self):
        for ent in self.doc.ents:
            if (ent.label_ == 'GPE') or (ent.label_ == 'LOC'):
                self.places.append(ent.text)
        if len(self.places) > 0:
            return ', '.join(list(set(self.places)))
        else:
            return ''

    def extract_orgs(self):
        for ent in self.doc.ents:
            if (ent.label_ == 'ORG') or (ent.label_ == 'FAC'):
                self.orgs.append(ent.text)
        if len(self.orgs) > 0:
            return ', '.join(list(set(self.orgs)))
        else:
            return ''

    def update_note(self):
        pass



class KeyBERTExtractor():
    def __init__(self, top_n=10, mmr=True, diversity=0.2, model='all-MiniLM-L6-v2'):
        self.kw_vectorizer = KeyphraseCountVectorizer()
        self.kw_extractor = KeyBERT(model=model)
        self.top_n = top_n
        self.mmr = mmr
        self.diversity = diversity
    def extract_keyphrases(self, doc, relevance=0.):
        doc = unidecode(doc)
        doc = re.sub(r'(\s+)',' ', doc)

        tags = self.kw_extractor.extract_keywords(docs=[doc], top_n=self.top_n, use_mmr=self.mmr,
                                                  diversity=self.diversity,
                                                  vectorizer=self.kw_vectorizer)
        tags = [x[0] for x in tags if x[1] >= relevance]
        return ','.join(tags)

    def update_note(self):
        pass