# Databricks notebook source
# MAGIC %md
# MAGIC ## CSRD directive

# COMMAND ----------

# MAGIC %md
# MAGIC On July 31, 2023, the European Commission adopted the [European Sustainability Reporting Standards](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202302772) (ESRS), which were published in the Official Journal of the European Union in December 2023. Drafted by the European Financial Reporting Advisory Group (EFRAG), the standards provide supplementary guidance for companies within the scope of the [E.U. Corporate Sustainability Reporting Directive](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022L2464) (CSRD). The adoption of the CSRD, along with the supporting ESRS, is intended to increase the breadth of nonfinancial information reported by companies and to ensure that the information reported is consistent, relevant, comparable, reliable, and easy to access.
# MAGIC
# MAGIC Source: [Deloitte](https://dart.deloitte.com/USDART/home/publications/deloitte/heads-up/2023/csrd-corporate-sustainability-reporting-directive-faqs)

# COMMAND ----------

# MAGIC %md
# MAGIC Though the CSRD compliance poses a data quality challenge to firms trying to collect and report this information for the first time, the directive itself (as per many regulatory documents) may be source of confusion / concerns and subject to interpretation. In this exercise, we want to demonstrate generative AI to navigate through the complexities of regulatory documents and the CSRD initiative specifically. We aim at programmatically extracting chapters / articles / paragraphs from the CSRD initiative (available below) and provide users with solid foundations to build GenAI solutions in the context of regulatory compliance.
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
# MAGIC
# MAGIC A second approach may be to read text as a whole and let generative AI capabilities extract specific sections and articles for us. Whilst this offer an ease of use and certainly in line with future direction of generative AI, we could possibly leave text at the interpretation of AI rather than relying on pure fact. 
# MAGIC
# MAGIC Instead, we went down the "boring" and "outdated" approach of scraping our document manually. Efforts done upfront will certainly pay off later when extracting facts around well defined chapter, articles, paragraphs and citations, acting as a compliance assistant through Q&A capabilities or operation workflow.

# COMMAND ----------

# MAGIC %md
# MAGIC We make use of the Beautiful soup library to navigate HTML content. 

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

# COMMAND ----------

import networkx as nx
import textwrap

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

import pandas as pd
node_df = pd.DataFrame(CSRD.nodes.data(), columns=['id', 'data'])
df = pd.json_normalize(node_df['data'])
df['id'] = node_df['id']
display(df[['id', 'group', 'label', 'title']])

# COMMAND ----------

from scripts.graph import displayGraph
displayHTML(displayGraph(CSRD))

# COMMAND ----------

from scripts.html_output import *
p_id = '3.9.7'
p = CSRD.nodes[p_id]
displayHTML(article_html(f'CSRD §{p_id}', p['title']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Indexing content

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

from langchain_community.embeddings import DatabricksEmbeddings
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

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

question = 'disclosing transactions between related parties, transactions between related parties included in a consolidation that are eliminated on consolidation shall not be included'

search, score = db.similarity_search_with_relevance_scores(question)[0]
displayHTML(vector_html('CSRD §{}'.format(search.metadata['id']), search.page_content, '{}%'.format(int(score * 100))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundational model

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 300, temperature=0)

# COMMAND ----------

# MAGIC %md
# MAGIC It is expected that foundational models like DBRX that learned knowledge from trillion of tokens may already know some of the relevant context for generic questions. However, it would be cavalier to ask specific questions to our model and solely rely on its foundational knowledge. In the context of regulatory compliance, we cannot afford for a model to "make things up" or return information as "best guess", regardless of how convincing its answer might be. The following question might yield well formed answer that may seem convincing to the naked eye, but definitely lacks substance and quality required.

# COMMAND ----------

query = 'Which disclosures will be subject to assurance, and what level of assurance is required?'
displayHTML(llm_html(query, chat_model.invoke(query).content))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## RAG strategy
# MAGIC Instead, we should prompt a model to search for specific knowledge provided as a context, knowledge that we acquired throughout the first part of our notebook. This creates our foundation for RAG framework. The main premise behind RAG is the injection of context (or knowledge) to the LLM in order to yield more accurate responses from it

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

question = {"query": 'Which disclosures will be subject to assurance, and what level of assurance is required?'}
answer = chain.invoke(question)
displayHTML(rag_html(question['query'], answer['result'], answer['source_documents']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracting references

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

from copy import deepcopy
CSRD_references = deepcopy(CSRD)
for i, x in reference_df.iterrows():
  if x['src_id'] != x['dst_id']:
    CSRD_references.add_edge(x['src_id'], x['dst_id'], color='coral')

# COMMAND ----------

displayHTML(displayGraph(CSRD_references))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compliance assistant

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

question = {"query": 'Which disclosures will be subject to assurance, and what level of assurance is required?'}
answer = chain_kg.invoke(question)
displayHTML(rag_kg_html(question['query'], answer['result'], answer['source_documents']))

# COMMAND ----------

# MAGIC %md
