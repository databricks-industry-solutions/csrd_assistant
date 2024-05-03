# Databricks notebook source
# MAGIC %md
# MAGIC ## Parsing CSRD directive

# COMMAND ----------

# MAGIC %md
# MAGIC We aim at programmatically extracting chapters / articles / paragraphs from the CSRD initiative (link below) and provide users with solid foundations to build more advanced GenAI applications in the context of regulatory compliance.
# MAGIC
# MAGIC https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02013L0034-20240109&qid=1712714544806

# COMMAND ----------

# MAGIC %run ./scripts/00_environment

# COMMAND ----------

import requests
act_url = 'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02013L0034-20240109&qid=1712714544806'
html_page = requests.get(act_url).text

# COMMAND ----------

# MAGIC %md
# MAGIC We may apply different data strategies to extract chapters and articles from the CSRD directive. The simplest approach would be to extract raw content and extract chunks that could feed our vector database. Whilst this would certainly be the easiest route, we would naively split text in the middle of potentially critical articles, not ensuring strict understanding of each sections. As a consequence, and even if we apply a context window across multiple chunks a model may be tempted to "infer" missing words and generate content not 100% in line with regulatory text.

# COMMAND ----------

# MAGIC %md
# MAGIC A second approach may be to read text as a whole and let generative AI capabilities extract specific sections and articles for us. Whilst this offer an ease of use and certainly in line with future direction of generative AI, we could possibly leave text at the interpretation of AI rather than relying on facts we could find as-is in the original documentation.

# COMMAND ----------

# MAGIC %md
# MAGIC Instead, we went down the "boring" and "outdated" approach of scraping documents manually. Efforts done upfront will certainly pay off later when extracting facts around chapters, articles, paragraphs and citations. We make use of the [Beautiful soup](https://beautiful-soup-4.readthedocs.io/en/latest/) library to navigate HTML content. Relatively complex, this HTML structure can be manually inspected through a browser / developer tool as per screenshot below.
# MAGIC
# MAGIC ![csrd_beautifulsoup.png](https://raw.githubusercontent.com/databricks-industry-solutions/csrd_assistant/main/images/csrd_beautifulsoup.png)

# COMMAND ----------

def get_directive_section(main_content):
  return main_content.find('div', {'class': 'eli-main-title'})

def get_content_section(main_content):
  return main_content.find('div', {'class': 'eli-subdivision'})

def get_chapter_sections(content_section):
  return content_section.find_all('div', recursive=False)

def get_article_sections(chapter_section):
  return chapter_section.find_all('div', {'class': 'eli-subdivision'}, recursive=False)

# COMMAND ----------

def get_directive_name(directive_section) -> str:
  title_doc = directive_section.find_all('p', {'class': 'title-doc-first'})
  title_doc = ' '.join([t.text.strip() for t in title_doc])
  return title_doc

def get_chapter_name(chapter_section) -> str:
  return chapter_section.find('p', {'class': 'title-division-2'}).text.strip().capitalize()

def get_chapter_id(chapter_section) -> str:
  chapter_id = chapter_section.find('p', {'class': 'title-division-1'}).text.strip()
  chapter_id = chapter_id.replace('CHAPTER', '').strip()
  return chapter_id

def get_article_name(article_section) -> str:
  return article_section.find('p', {'class': 'stitle-article-norm'}).text.strip()

def get_article_id(article_section) -> str:
  article_id = article_section.find('p', {'class': 'title-article-norm'}).text.strip()
  article_id = re.sub('\"?Article\s*', '', article_id).strip()
  return article_id

# COMMAND ----------

from bs4.element import Tag
import re

def _clean_paragraph(txt):
  # remove multiple break lines
  txt = re.sub('\n+', '\n', txt)
  # simplifies bullet points
  txt = re.sub('(\([\d\w]+\)\s?)\n', r'\1\t', txt)
  # simplifies quote
  txt = re.sub('‘', '\'', txt)
  # some weird references to other articles
  txt = re.sub('\(\\n[\d\w]+\n\)', '', txt)
  # remove spaces before punctuation
  txt = re.sub(f'\s([\.;:])', r'\1', txt)
  # remove reference links
  txt = re.sub('▼\w+\n', '', txt)
  # format numbers
  txt = re.sub('(?<=\d)\s(?=\d)', '', txt)
  # remove consecutive spaces
  txt = re.sub('\s{2,}', ' ', txt)
  # remove leading / trailing spaces
  txt = txt.strip()
  return txt 

def get_paragraphs(article_section):
  content = {}
  paragraph_number = '0'
  paragraph_content = []
  for child in article_section.children:
    if isinstance(child, Tag):
      if 'norm' in child.attrs.get('class'):
        if child.name == 'p':
          paragraph_content.append(child.text.strip())
        elif child.name == 'div':
          content[paragraph_number] = _clean_paragraph('\n'.join(paragraph_content))
          paragraph_number = child.find('span', {'class': 'no-parag'}).text.strip().split('.')[0]
          paragraph_content = [child.find('div', {'class': 'inline-element'}).text]
      elif 'grid-container' in child.attrs.get('class'):
        paragraph_content.append(child.text)
    content[paragraph_number] = _clean_paragraph('\n'.join(paragraph_content))
  return {k:v for k, v in content.items() if len(v) > 0}

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we could extract the full content hierarchy from the CSRD directive, from chapter to articles and paragraph.

# COMMAND ----------

from bs4 import BeautifulSoup

main_content = BeautifulSoup(html_page, 'html.parser')
directive_section = get_directive_section(main_content)
directive_name = get_directive_name(directive_section)
content_section = get_content_section(main_content)

for chapter_section in get_chapter_sections(content_section):
  chapter_id = get_chapter_id(chapter_section)
  chapter_name = get_chapter_name(chapter_section)
  articles = len(get_article_sections(chapter_section))
  print(f'Chapter {chapter_id}: {chapter_name}')
  print(f'{articles} article(s)')
  print('')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Knowledge Graph
# MAGIC Our content follows a tree structure where each chapter has multiple articles and each article has multiple paragraphs. A graph structure becomes a logical representation of our data. We leverage [NetworkX](https://networkx.org/) libary for that purpose.

# COMMAND ----------

import networkx as nx

CSRD = nx.DiGraph()
CSRD.add_node('CSRD', title=directive_name, label='CSRD', group='DIRECTIVE')

for chapter_section in get_chapter_sections(content_section):
  chapter_id = get_chapter_id(chapter_section)
  chapter_name = get_chapter_name(chapter_section)

  CSRD.add_node(f'{chapter_id}', title=chapter_name, label=f'Chapter {chapter_id}', group='CHAPTER')
  CSRD.add_edge('CSRD', f'{chapter_id}')

  for article_section in get_article_sections(chapter_section):
    article_id = get_article_id(article_section)
    article_name = get_article_name(article_section)
    article_paragraphs = get_paragraphs(article_section)

    CSRD.add_node(f'{chapter_id}.{article_id}', title=article_name, label=f'Article {article_id}', group='ARTICLE')
    CSRD.add_edge(chapter_id, f'{chapter_id}.{article_id}' )

    for paragraph_id, paragraph_text in article_paragraphs.items():
      CSRD.add_node(f'{chapter_id}.{article_id}.{paragraph_id}', title=paragraph_text, label=f'Article {article_id}({paragraph_id})',group='PARAGRAPH' )
      CSRD.add_edge(f'{chapter_id}.{article_id}', f'{chapter_id}.{article_id}.{paragraph_id}')

# COMMAND ----------

# MAGIC %md
# MAGIC We can easily access any given document in our graph and manually investigate its content, further validating our parsing logic earlier. 

# COMMAND ----------

import pandas as pd
node_df = pd.DataFrame(CSRD.nodes.data(), columns=['id', 'data'])
df = pd.json_normalize(node_df['data'])
df['id'] = node_df['id']
display(df[['id', 'group', 'label', 'title']])

# COMMAND ----------

# MAGIC %md
# MAGIC Stored as a graph, the same can easily be visualized to get a better understanding of the problem at hand. Our directive contains ~ 350 nodes where each node is connected to maximum 1 parent (expected from a tree structure), as represented below. Zoom in and hover some nodes to access their actual text content.

# COMMAND ----------

from scripts.graph import displayGraph
displayHTML(displayGraph(CSRD))

# COMMAND ----------

# MAGIC %md
# MAGIC We created our graph identifiers so that we can access a given paragraph through their unique ID, expressed in the form of `chapter-article-paragraph` coordinate.

# COMMAND ----------

from scripts.html_output import *
p_id = '3.9.7'
p = CSRD.nodes[p_id]
displayHTML(article_html(f'CSRD §{p_id}', p['title']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Indexing content
# MAGIC Though representing our CSRD as a graph was visually compelling, it offers no semantic search capability. In this section, we will further index our graph data to offer search functionality for a given question / query. 

# COMMAND ----------

csrd_data = []
for node_id, node_data in CSRD.nodes.data():
  if node_data['group'] == 'PARAGRAPH':
    node_label = node_data.get('label')
    node_text = node_data.get('title')
    xs = node_id.split('.')

    node_content = f'{node_label}:\n{node_text}'
    csrd_data.append(pd.Series({
      'id': node_id,
      'chapter': xs[0],
      'article': xs[1],
      'paragraph': xs[2],
      'content': node_content
    }))

csrd_df = pd.DataFrame(csrd_data)
display(csrd_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In production settings, we highly encourage users to leverage [Databricks vector store](https://docs.databricks.com/en/generative-ai/vector-search.html) capability, linking records down to the actual binary file that may have been previously stored on your volume, hence part of a unified governance strategy. 
# MAGIC
# MAGIC In the context of a solution accelerator, we decided to limit the infrastructure requirements (volume creation, table creation, DLT pipelines, etc.) and provide entry level capabilities, in memory, leveraging FAISS as our vector store and leveraging out-of-the-box foundation models provided by Databricks.
# MAGIC
# MAGIC For more information on E2E applications, please refer to [DB Demo](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot).

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

# MAGIC %md
# MAGIC We leverage langchain framework for its ease of use, converting our graph content as a set of langchain documents.

# COMMAND ----------

from langchain.docstore.document import Document

documents = []
for i, rec in csrd_df.iterrows():
  metadata = rec.to_dict()
  page_content=metadata['content']
  del metadata['content']
  documents.append(Document(
    page_content=page_content,
    metadata=metadata
  ))

# COMMAND ----------

from langchain.vectorstores import FAISS
db = FAISS.from_documents(documents=documents, embedding=embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now retrieve content based on similarity search. Given a question, part of a text, or simple keywords, we retrieve specific facts, articles and paragraph that we can trace back to our CSRD directive. This will become the foundation to our RAG strategy later. The example below returns the best matching paragraph with a relevance score given a user question.

# COMMAND ----------

question = 'disclosing transactions between related parties, transactions between related parties included in a consolidation that are eliminated on consolidation shall not be included'

search, score = db.similarity_search_with_relevance_scores(question)[0]
displayHTML(vector_html('CSRD §{}'.format(search.metadata['id']), search.page_content, '{}%'.format(int(score * 100))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundational model
# MAGIC It is expected that foundational models like DBRX, Llama 3, Mixtral or OpenAI that learned knowledge from trillion of tokens may already know some of the relevant context for generic questions. However, it would be cavalier to ask specific questions to our model and solely rely on its foundational knowledge. In the context of regulatory compliance, we cannot afford for a model to "make things up" or return information as "best guess", regardless of how convincing its answer might be. 

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 300, temperature=0)

# COMMAND ----------

query = 'Which disclosures will be subject to assurance, and what level of assurance is required?'
displayHTML(llm_html(query, chat_model.invoke(query).content))

# COMMAND ----------

# MAGIC %md
# MAGIC The question above might yield a well formed answer that may seem convincing to the naked eye, but definitely lacks substance and the quality required for regulatory compliance. Instead of a POV we may take at its face value, it would certainly be comfortable to cite actual facts and reference existing articles / paragraphs from a trusted source (such as the directive itself).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## RAG strategy
# MAGIC Instead, we should prompt a model to search for specific knowledge, knowledge that we acquired throughout the first part of our notebook. This creates our foundation for RAG. In this section, we will design a simple prompt and "chain" our vector store logic with actual model inference.

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
import json

TEMPLATE = """
Context information is below.

---------------------
{context}
---------------------

Given the context information and not prior knowledge.
Answer compliance issue related to the CSRD directive only.

If the question is not related to regulatory compliance, kindly decline to answer. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible, citing articles and chapters whenever applicable.
Please do not repeat the answer and do not add any additional information. 

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents = True
)

# COMMAND ----------

# MAGIC %md
# MAGIC Prompts must be designed in a way that provides user with better clarity and / or confidence as well as safeguarding model against malicious or offensive use. This, however, is not part of this solution but invite users to explore our [DB Demo](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot) that covers the basics to most advanced use of RAG. A good example would be to restrict our model to only answer questions that are CSRD related, possibly linking multiple chains together.

# COMMAND ----------

question = {"query": 'Which disclosures will be subject to assurance, and what level of assurance is required?'}
answer = chain.invoke(question)
displayHTML(rag_html(question['query'], answer['result'], answer['source_documents']))

# COMMAND ----------

# MAGIC %md
# MAGIC In this example, we let our model formulate a point of view based on actual facts we can trust. For the purpose of that demo, we represented output as a form of a notebook. In real life scenario, offering that capability as a chat interface would require building a UI and application server outside of a notebook based environment (outside of the scope here). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracting references
# MAGIC It is expected that regulatory documents (or legal documents) might contain multiple definitions and cross references to other articles, paragraphs or other regulations. Whilst we kept the scope of this demo to the CSRD initiative only, our document might already contain many cross references that would be needed to formulate an objective view with objective facts.

# COMMAND ----------

prompt_ref = """
Context information is below.

---------------------
{text}
---------------------

Here is an excerpt of a regulatory article that may contain references to other articles. 
Extract references to article and paragraphs explicitly mentioned in that text. Do not infer additional references besides those being explicitly mentioned.
 
If the text does not specify the article number, use article {article}.
If the text does not specify the paragraph number, use paragraph 0.
Return all references in a format [article_number]-[paragraph_number]. Add justification.

Answer:
"""

# COMMAND ----------

# MAGIC %md
# MAGIC This is a perfect example where large language model reasoning capabilities might shine. By designing a simple prompt and parsing logic, one can extract references to other articles and paragraph we know exist in our graph structure.

# COMMAND ----------

def parse_references(txt, graph):
  node_ids = []
  references = re.findall('(\d+[a-z]?)\-(\d+[a-z]?)', txt)
  for reference in references:
    article_id = reference[0]
    paragraph_id = reference[1]
    if paragraph_id == '0':
      # Retrieve all paragraphs in a given article
      node_ids.extend(list(filter(lambda x: re.match(f'\d+\.{article_id}\..*', x), list(graph.nodes))))
    else:
      # Retrieve a given paragraph
      node_ids.extend(list(filter(lambda x: re.match(f'\d+\.{article_id}\.{paragraph_id}', x), list(graph.nodes))))
  return node_ids

# COMMAND ----------

chapter='6'
article='29'
paragraph='2'
test_node=CSRD.nodes[f'{chapter}.{article}.{paragraph}']
answer = chat_model.invoke(prompt_ref.format(article=article, text=test_node['title'])).content
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC Modern foundational models such as DBRX are able to follow our prompting instructions, returning references to other articles we know exist. In the example below, we return cross references to Chapter 6, article 29 and paragraph 2.

# COMMAND ----------

references = parse_references(answer, CSRD)
displayHTML(references_html('CSRD §{}'.format(f'{chapter}.{article}.{paragraph}'), test_node['title'], references))

# COMMAND ----------

reference_data = []

for node_id, node_data in CSRD.nodes.data():
  if node_data['group'] == 'PARAGRAPH':
    article_id = node_id.split('.')[1]
    answer = chat_model.invoke(prompt_ref.format(article=article_id, text=node_data['title'])).content
    references = parse_references(answer, CSRD)
    reference_data.append([node_id, references, answer, node_data['title']])

display(pd.DataFrame(reference_data, columns=['node_id', 'references', 'justification', 'content']))

# COMMAND ----------

reference_df = pd.DataFrame(reference_data, columns=['node_id', 'references', 'justification', 'content'])
reference_df = reference_df.explode('references')
reference_df = reference_df.rename({'references': 'dst_id', 'node_id': 'src_id'}, axis=1)
reference_df = reference_df[['src_id', 'dst_id']]
reference_df = reference_df.dropna()
display(reference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Unfortunately for us, regulatory documents are complex and show high number of connections between different paragraphs and articles. We might want to visualize the resulting graph to grasp its newly acquired complexity.

# COMMAND ----------

from copy import deepcopy
CSRD_references = deepcopy(CSRD)
for i, x in reference_df.iterrows():
  if x['src_id'] != x['dst_id']:
    CSRD_references.add_edge(x['src_id'], x['dst_id'], color='coral')

# COMMAND ----------

displayHTML(displayGraph(CSRD_references))

# COMMAND ----------

# MAGIC %md
# MAGIC We directly get a sense of the regulatory complexity of the CSRD directive. Each "clique" represented here may be source of confusion for whoever does not have a legal background. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compliance assistant
# MAGIC In this section, we want to leverage our newly acquired knowledge of article references to further augment the accuracy and relevance of a regulatory assistant. Being able to traverse our graph and navigating through references and definitions might present opportunities for businesses, better adhering to regulatory guideline by further understanding its legal terms. A first approach might be to expand the capabilities of a vector store by linking content to its actual connections.

# COMMAND ----------

from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores.base import VectorStoreRetriever
from typing import List
from networkx.classes.digraph import DiGraph

class CustomRetriever(BaseRetriever):

  retriever: VectorStoreRetriever
  knowledge_graph: DiGraph

  def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

    # Use existing retriever to get the documents
    documents = self.retriever.get_relevant_documents(query, callbacks=run_manager.get_child())

    # Retrieve document Ids
    doc_ids = [doc.metadata['id'] for doc in documents]

    # Retrieve nodes
    nodes = [[node_id, self.knowledge_graph.nodes.get(node_id)] for node_id in doc_ids]
    nodes = [[node_id, node_data] for node_id, node_data in nodes if node_data is not None]

    # Build documents in relevance order
    processed_ids = set()
    supporting_documents = []
    for node_id, node_data in nodes:
      node_content = '{}:\n{}'.format(node_data['label'], node_data['title'])
      if node_data['group'] == 'PARAGRAPH' and node_id not in processed_ids:
        processed_ids.add(node_id)
        supporting_documents.append(Document(page_content=node_content, metadata={'id': node_id}))

      # Traverse graph to get cross reference articles
      children_id = list(self.knowledge_graph.successors(node_id))
      for child_id in children_id:
        child_data = self.knowledge_graph.nodes[child_id]
        node_content = '{}:\n{}'.format(child_data['label'], child_data['title'])
        if child_data['group'] == 'PARAGRAPH' and child_id not in processed_ids:
          processed_ids.add(child_id)
          supporting_documents.append(Document(page_content=node_content, metadata={'id': child_id}))

    return supporting_documents

# COMMAND ----------

chain_kg = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=CustomRetriever(retriever=db.as_retriever(), knowledge_graph=CSRD_references),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents = True
)

# COMMAND ----------

# MAGIC %md
# MAGIC In the example below, the same question was triggering additional content retrieval that could be used to return a more objective and accurate information.

# COMMAND ----------

question = {"query": 'Which disclosures will be subject to assurance, and what level of assurance is required?'}
answer = chain_kg.invoke(question)
displayHTML(rag_kg_html(question['query'], answer['result'], answer['source_documents']))

# COMMAND ----------

# MAGIC %md
# MAGIC The downside, however, is that our graph is highly connected, resulting in far too much content being returned (even when limiting traversal to maximum 1 hop). Though modern models can handle larger context window (32k tokens for DBRX), a model might not be able to fully exploit each information returned (tend to ignore context not at the beigining or end of the context window). An alternative scenario would be to probe our vector store recursively, formulating an answer by ensuring full comprehension of each of the returned definitions. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi stage reasoning
