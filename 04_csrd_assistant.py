# Databricks notebook source
# MAGIC %md
# MAGIC ## Compliance assistant
# MAGIC In this section, we want to leverage our newly acquired knowledge of articles and referenced articles to further augment the accuracy and relevance of any Generative AI powered regulatory assistant. Being able to traverse a knowledge graph and navigating through legal references and definitions might present opportunities for many consultancy businesses to improve regulatory compliance and reduce operation expenses of their clients.

# COMMAND ----------

# MAGIC %run ./config/00_environment

# COMMAND ----------

# MAGIC %md
# MAGIC We load our vector store built in our previous notebook. Once again, we cannot further state the importance of leveraging governed tables and vector store capabilities rather than in memory libraries like in this demo (note that `allow_dangerous_deserialization` used here). 

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

from langchain.vectorstores import FAISS
CSRD_search = FAISS.load_local(faiss_output_dir, embeddings=embeddings, allow_dangerous_deserialization=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Similarly, we load our graph object by first reading our delta tables of nodes and edges.

# COMMAND ----------

nodes_df = spark.read.table(table_nodes).toPandas()
edges_df = spark.read.table(table_edges).toPandas()

# COMMAND ----------

import networkx as nx
CSRD_graph = nx.DiGraph()
for i, n in nodes_df.iterrows():
  CSRD_graph.add_node(n['id'], label=n['label'], title=n['content'], group=n['group'])
for i, e in edges_df.iterrows():
  CSRD_graph.add_edge(e['src'], e['dst'], label=e['label'])

# COMMAND ----------

# MAGIC %md
# MAGIC In a previous section, we showed how langchain could help us "chain" our model with our vector store as part of a RAG strategy. In this section, we will extend the `BaseRetriever` class to further expand our search to relevant nodes and its referenced content (limiting our search to 1-hop in our graph)

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 300, temperature=0)

# COMMAND ----------

from langchain.schema.retriever import BaseRetriever
from langchain.docstore.document import Document
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
      if node_data['group'] == 'PARAGRAPH' and node_id not in processed_ids:
        processed_ids.add(node_id)
        supporting_documents.append(Document(page_content=node_data['title'], metadata={'id': node_id, 'label': node_data['label']}))

      # Traverse graph to get cross reference articles
      children_id = list(self.knowledge_graph.successors(node_id))
      for child_id in children_id:
        child_data = self.knowledge_graph.nodes[child_id]
        if child_data['group'] == 'PARAGRAPH' and child_id not in processed_ids:
          processed_ids.add(child_id)
          supporting_documents.append(Document(page_content=child_data['title'], metadata={'id': child_id, 'label': child_data['label']}))

    return supporting_documents

# COMMAND ----------

# MAGIC %md
# MAGIC Our prompt remain similar as per our previous example, returning articles as part of our model context.

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

# COMMAND ----------

chain_kg = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=CustomRetriever(retriever=CSRD_search.as_retriever(), knowledge_graph=CSRD_graph),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents = True
)

# COMMAND ----------

# MAGIC %md
# MAGIC As reported below, a simple question has now triggered additional content search that could be used to return a more objective and accurate answer.

# COMMAND ----------

from scripts.html_output import *

question = {"query": 'Which disclosures will be subject to assurance, and what level of assurance is required?'}
answer = chain_kg.invoke(question)
displayHTML(rag_kg_html(question['query'], answer['result'], answer['source_documents']))

# COMMAND ----------

# MAGIC %md
# MAGIC Although we brought relevant content in order to formulate a more objective answer, too much content is being returned (even when limiting traversal to maximum 1 hop) because of the high number of connection in our knowledge graph. Though modern models can handle larger context window (32k tokens for DBRX), a model might not be able to equally exploit each information returned (model tend to ignore context not at the begining or end of the context window) and / or might simply fail because of context size. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi stage reasoning
# MAGIC An alternative scenario to the above would be to probe our vector store recursively, formulating an answer by ensuring full comprehension of each of the returned definitions, iteratively. This can be addressed through multi stage reasoning by defining multiple agents. At the time of this demo, OpenAI model my exhibit higher degree of reasoning than most open source models. For the purpose of that demo, we will leverage OpenAI API by loading our private key through databricks [secrets](https://docs.databricks.com/en/security/secrets/index.html).

# COMMAND ----------

from langchain_openai import ChatOpenAI
import os

# COMMAND ----------


os.environ["OPENAI_API_KEY"] = dbutils.secrets.get('industry-solutions', 'openai_key')
model = ChatOpenAI(temperature=0.05, model_name='gpt-3.5-turbo', max_tokens=500)

# COMMAND ----------

from pydantic import BaseModel, Field
from langchain.agents import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.agent import AgentExecutor

# COMMAND ----------

# MAGIC %md
# MAGIC We will define two agents. Whilst the former will be responsible from reading our vector store for relevant content, the latter might be called to fetch referenced article content, enriching prompt to let our model acquire all its necessary knowledge. At the time of this demo, agent modelling is still an active area of research and might be considered a black box with respect to calling individual tools / functions. We maximize the relevance of our tools by adding the right level of documentation.

# COMMAND ----------

class SearchReferenceByArticleId(BaseModel):
  article_reference: str = Field(..., description="The referenced article for which you need further information.")

@tool('expand_search_reference', args_schema=SearchReferenceByArticleId)
def search_reference(article_reference: str) -> str:
  """Use this tool when you need to search for additional articles referenced in the article content."""

  hit = CSRD_graph.nodes[article_reference]
  doc_references = ','.join([d for d in list(CSRD_graph.neighbors(article_reference)) if d != article_reference])
  doc_content = hit['title']
  doc_label = hit['label']
  response = f'''###
[article_id]: {article_reference}
[article_name]: {doc_label}
[article_references]: {doc_references}
[article_content]: {doc_content}'''
  return response

# COMMAND ----------

class FindArticleByQuerySimilarity(BaseModel):
  query: str = Field(..., description="The question or query for which you need additional information.")

@tool('search_content', args_schema=FindArticleByQuerySimilarity)
def search_content(query: str) -> str:
  """Use this tool to search for content that is relevant to a given question."""

  hits = CSRD_search.similarity_search_with_relevance_scores(query)
  response = []
  for doc, score in hits:
    doc_id = doc.metadata['id']
    doc_label = doc.metadata['label']
    doc_content = doc.page_content
    doc_references = ','.join([d for d in list(CSRD_graph.neighbors(doc_id)) if d != doc_id])
    response.append(f'''###
[article_id]: {doc_id}
[article_name]: {doc_label}
[article_references]: {doc_references}
[article_content]: {doc_content}''')
    
  response = "\n\n".join(response)
  return response

# COMMAND ----------

tools = [search_reference, search_content]
functions = [convert_to_openai_function(f) for f in tools]

# COMMAND ----------

instructions = """Given the context information, answer compliance issues related to the CSRD directive.
Start your search with content related to a given query using the [search_reference] tool. 
Each article may have [article_references] to other articles. Expand your search using the [expand_search_reference] tool.
Continue your search until all referenced information have been used to answer the question.

If the question is not related to regulatory compliance, kindly decline to answer. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Return concise information answering the question and citing all the relevant [article_name]."""

prompt = ChatPromptTemplate.from_messages([
  ("system", instructions),
  ("user", "{input}"),
  MessagesPlaceholder(variable_name="scratchpad")
])

# COMMAND ----------

scratch_pad = RunnablePassthrough.assign(scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"]))

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we can chain our model that we bind with our couple of functions and a prompt that we designed for that specific scenario.

# COMMAND ----------

chain = scratch_pad | prompt | model.bind(functions=functions) | OpenAIFunctionsAgentOutputParser()
agent = AgentExecutor(
  agent=chain,
  tools=tools,
  verbose=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC By adding `verbose` parameter to our agent, we can get a glimpse at the model reasoning capability. 

# COMMAND ----------

query = 'List all the conditions whereby a company structure is allowed to limit their reporting to business strategy only.'
answer = agent.invoke({"input": query})

# COMMAND ----------

displayHTML(llm_html(query, answer['output']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Closing thoughts
# MAGIC We appreciate we barely scratched the surface of multi stage reasoning. We invite users to go through further documentation such as the excellent [knowledge graphs rag short course](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) on deeplearning.ai. However, through this solution, we proved the strategic relevance of generative AI capabilities and the importance of bringing RAG applications to the complex field of regulatory compliance. By expanding the scope beyond CSRD, to multiple regulatory documents, one may consider fine tuning a model and better understand regulatory structures across multiple markets / segments / industries.

# COMMAND ----------

query = 'Quelles sont les conditions pur que mon entreprise soit sujette au statut de micro entreprise? Dans le case de micro enterprise, quelles sont les dispositions particulieres en termes de normes de reporting?'

answer = agent.invoke({"input": query})
displayHTML(llm_html(query, answer['output']))

# COMMAND ----------

# MAGIC %md
# MAGIC As a final thought, one may wonder how the same applies to different languages than english. Running the same in French should hopefully give compliance officer a better idea of the opportunities generative AI capabilities can bring to the future of risk and compliance.
