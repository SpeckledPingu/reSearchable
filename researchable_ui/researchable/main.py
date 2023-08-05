import streamsync as ss
import pandas as pd
import json
import numpy as np
import requests
import spacy

nlp = spacy.load("en_core_web_sm")

# This is a placeholder to get you started or refresh your memory.
# Delete it or adapt it as necessary.
# Documentation is available at https://streamsync.cloud

# Shows in the log when the app starts
print("Hello world!")

def query(state):
    query = state['query']
    results = requests.get('http://localhost:10100/query/sparse',
                            params={'query':query})
    results = json.loads(results.text)
    print(results[0].keys())
    results = {str(i):_values for i, _values in enumerate(results[:5])}
    for k, v in results.items():
        v['truncated_content'] = v['content'][:1000] + '...'
    state['query_results'] = results
    print(results.keys())
def _display_results(state, context):
    pass

def clear_notes(state):
    # clears all the note fields
    pass

def auto_notes(state):
    # call llm to summarize
    results = requests.post('http://localhost:10101/ask/summary',  params={'doc':state['article_text']})
    print(results.text)
    state['notes'] = json.loads(results.text)['summary']

def auto_tag(state):
    # call keybert to extract keyphrases
    results = requests.post('http://localhost:10101/ask/tag',  params={'doc':state['article_text']})
    print(results.text)
    state['notes_tags'] = ', '.join(json.loads(results.text)['tags'])

def auto_people(state):
    # call spacy to NER people
    doc = nlp(state['article_text'])
    people = list()
    for ent in doc.ents:
        if (ent.label_ == 'PERSON') or (ent.label_ == 'NORP'):
            people.append(ent.text)
    print(people)
    if len(people) > 0:
        state['notes_people'] = ', '.join(list(set(people)))
        

def auto_places(state):
    doc = nlp(state['article_text'])
    places = list()
    for ent in doc.ents:
        if (ent.label_ == 'GPE') or (ent.label_ == 'LOC'):
            places.append(ent.text)
    print(places)
    if len(places) > 0:
        state['notes_places'] = ', '.join(list(set(places)))

def auto_orgs(state):
    # call spacy to NER orgs
    doc = nlp(state['article_text'])
    orgs = list()
    for ent in doc.ents:
        if (ent.label_ == 'ORG') or (ent.label_ == 'FAC'):
            orgs.append(ent.text)
    print(orgs)
    if len(orgs) > 0:
        state['notes_orgs'] = ', '.join(list(set(orgs)))

def auto_dates(state):
    # call spacy to NER orgs
    doc = nlp(state['article_text'])
    dates = list()
    for ent in doc.ents:
        if (ent.label_ == 'DATE') or (ent.label_ == 'TIME'):
            dates.append(ent.text)
    print(dates)
    if len(dates) > 0:
        state['notes_dates'] = ', '.join(list(set(dates)))

def payload_inspector(state, context):
    # Shown every time the event handler is executed
    # print("Payload: " + repr(context))
    state['article_title'] = context['item']['title']
    state['article_publication'] = context['item']['publication']
    state['article_text'] = context['item']['content']
    state.set_page("annotate")

# The following code will set the value of product_id
# to the value of the "product" state element
# def change_route_vars(state):
#     state.set_route_vars({
#         "doc_id": state["doc_id"]
#     })

# The following event handler reads the product_id route var,
# then assigns its value to the "product" state element.
# def handle_hash_change(state, payload):
#     route_vars = payload.get("route_vars")
#     if not route_vars:
#         return
#     state["doc_id"] = route_vars.get("doc_id")

initial_state = ss.init_state({
    "my_app": {
        "title": "My App"
    },
    "typesense_url":"localhost:10100",
    'hybrid_endpoint':'/query/hybrid_rerank',
    'dense_endpoint':'/query/dense',
    'document_endpoint':'/query/by_id',
    'ner_url':'localhost:10101',
    'ner_endpoint':'/extract_all',
    'query':'',
    'query_results':{'0':{'content':' ','title':' ','publication':' '}}
})

