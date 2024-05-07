# Databricks notebook source
# MAGIC %md
# MAGIC ## Cross references
# MAGIC It is expected that regulatory documents (or legal documents in general) might contain multiple definitions and cross references to other articles, paragraphs, annex or other regulations. Whilst we kept the scope of this demo to the CSRD initiative only, our document might already contain many cross references that would be needed to formulate an objective view with objective facts. In this section, we will extract all references and complement our knowledge graph with further references and definitions.

# COMMAND ----------

# MAGIC %run ./config/00_environment

# COMMAND ----------

nodes_df = spark.read.table(table_nodes).toPandas()
edges_df = spark.read.table(table_edges).toPandas()

# COMMAND ----------

import networkx as nx

CSRD = nx.DiGraph()

for i, n in nodes_df.iterrows():
  CSRD.add_node(n['id'], label=n['label'], title=n['content'], group=n['group'])

for i, e in edges_df.iterrows():
  if e['label'] == 'CONTAINS':
    CSRD.add_edge(e['src'], e['dst'], label=e['label'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prompt engineering
# MAGIC This is a perfect example where LLM reasoning capabilities might shine. By combining a simple prompt and parsing logic, one can extract references to other articles and paragraph we know exist in our graph structure.

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

import re

def parse_references(txt, graph):
  node_ids = []
  references = re.findall('(\d+[a-z]?)\-(\d+[a-z]?)', txt)
  for reference in references:
    article_id = reference[0]
    paragraph_id = reference[1]
    if paragraph_id == '0':
      # Retrieve all paragraphs in a given article
      node_ids.extend(list(filter(lambda x: re.match(f'^\d+\.{article_id}\..*$', x), list(graph.nodes))))
    else:
      # Retrieve a given paragraph
      node_ids.extend(list(filter(lambda x: re.match(f'^\d+\.{article_id}\.{paragraph_id}$', x), list(graph.nodes))))
  return node_ids

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting references
# MAGIC We delegate that task to our foundation model available out of the box on our databricks workspace.

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 300, temperature=0)

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

from scripts.html_output import *
references = parse_references(answer, CSRD)
displayHTML(references_html(test_node['label'], test_node['title'], references))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's extend this logic to every node in our knowledge graph

# COMMAND ----------

reference_data = []

for node_id, node_data in CSRD.nodes.data():
  if node_data['group'] == 'PARAGRAPH':
    article_id = node_id.split('.')[1]
    answer = chat_model.invoke(prompt_ref.format(article=article_id, text=node_data['title'])).content
    references = parse_references(answer, CSRD)
    reference_data.append([node_id, references, 'REFERENCES'])

# COMMAND ----------

import pandas as pd
reference_df = pd.DataFrame(reference_data, columns=['src', 'dst', 'label'])
reference_df = reference_df.explode('dst')
reference_df = reference_df.dropna()
display(reference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Unfortunately (but expectedly), regulatory documents are complex and show high number of references between different paragraphs and articles. We append our changes as new edges in our table.

# COMMAND ----------

_ = spark.createDataFrame(reference_df).write.format('delta').mode('append').saveAsTable(table_edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Updated graph
# MAGIC We might want to visualize the resulting graph to grasp its newly acquired complexity. We load our original graph with new appended edges, color coded in red.

# COMMAND ----------

nodes_df = spark.read.table(table_nodes).toPandas()
edges_df = spark.read.table(table_edges).toPandas()

# COMMAND ----------

import networkx as nx

CSRD_references = nx.DiGraph()

for i, n in nodes_df.iterrows():
  CSRD_references.add_node(n['id'], label=n['label'], title=n['content'], group=n['group'])

for i, e in edges_df.iterrows():
  if e['src'] != e['dst']:
    if e['label'] == 'REFERENCES':
      CSRD_references.add_edge(e['src'], e['dst'], color='coral')
    else:
      CSRD_references.add_edge(e['src'], e['dst'])

# COMMAND ----------

from scripts.graph import displayGraph
displayHTML(displayGraph(CSRD_references))

# COMMAND ----------

# MAGIC %md
# MAGIC Using a simple visualization, we get a sense of the complexity behind the CSRD directive. Each "clique" (i.e. highly connected hub) represented here may be source of confusion or dispute for whoever does not have a legal background, possibly explaining why so many organizations recently started to offer CSRD specific consultancy practices.
