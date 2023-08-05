# reSearchable
Researchable is a toolkit for setting up local research and note taking for journalists, students, and anyone else who sifts through large amounts of data.

Most research involves large amounts of text. This may be scraped text, pdfs, and other sources that need to be processed and then indexed into a flexible search engine.

It is motivated by ensuring several concepts are maintained with one overriding principle:

**LLMs are assistants, not replacements**

**First: It should be able to be fully local.**

All the LLMs, apis, and search engine for them should be hostable on a local machine.

**Second: Parts of the platform should be cloud-ifiable.**

If an index for documents is too big for local hosting, it should be movable to the cloud. Same with LLMs and any other add ons.

**Third: The index needs to be fully featured.**

Typesense is used as the baseline, but existing indexes people have access to can include Solr, Elastic, OpenSearch, and others. 
Don't reinvent the wheel. If your data is already hosted, you should use that.

Simple vector stores (e.g. Annoy) and BM25 stores have little advantages beyond their vanilla relevance scoring.
Typesense and mature indexes allow faceting, fields, and more complex relevance and search capabilities.
To move from just vectors or lexical search, the indexes need to be improved to improve research.

**Fourth: LLMs don't do all the work.**

While ChatGPT, Claude, and others will be hooked up, the LLMs won't process text and force tinkering with prompt engineering.

Autogenerated notes, summaries, and extractions should be transparent, individually selectable, and editable before being saved.

Rather than relying on external LLMs alone, this project aims to have a fully local suite of LLMs to assist in extractions, summaries, and searching.
The search index needs to be local and powerful. The problem with basic BM25 and vector search systems is the difficulty in scaling and managing them.
Ideally, a system should have an index to be hosted locally (e.g. Typesense in Docker) or in the cloud, and perform lexical and vector search.

**Fifth: Your notes should be interactive and explorable**

This next step is in progress, but notes should be leveraged with additional searches to find what you previously looked at. 
Extractions and notes should be stored as both documents (CouchDB/Typesense) and as a graph (NetworkX/Neo4J).

**Future Work**

Your notes should let you fine tune or develop the LLM tools to customize it for your organization or personal use on the research project.
After large amounts of research, you've developed an idea of what is important and what isn't. This should be stored and usable to then tune
the LLMs to assist in more research, or help others search the topic.
