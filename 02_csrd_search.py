# Databricks notebook source
# MAGIC %md
# MAGIC ## Indexing content
# MAGIC Though representing our CSRD directive as a graph is visually compelling, it offers little to no search capability. We further index our graph data to offer semantic search functionality for a given question that can be combined with Generative AI capabilities as part of a Retrieval Augmented Generation strategy (RAG). We appreciate that leveraging [Databricks vector store](https://docs.databricks.com/en/generative-ai/vector-search.html) in the context of this solution might sound an overkill relative to in memory options like ChromaDB or FAISS. However, we encourage users to leverage this capability in production settings where scope might go beyond the CSRD directive only and where production quality content and operation governance are critical to the success of this initiative in the harsh field of regulatory compliance.

# COMMAND ----------

# MAGIC %run ./config/00_environment

# COMMAND ----------

nodes_df = spark.read.table(table_nodes).toPandas()
display(nodes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We maintained high quality text content as operation table that can be streamed (continuously or triggered) down to our vector store (abstracted through the `synchronize_vector_index` method here). We make use of the langchain library to further abstract our vector endpoint logic away and ensure full compatibility with other langchain applications.

# COMMAND ----------

vector_index = synchronize_vector_index()

# COMMAND ----------

from langchain_community.vectorstores import DatabricksVectorSearch

CSRD_search = DatabricksVectorSearch(
  vector_index, 
  text_column="content", 
  columns=["id", "label"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC With our index store up to date with latest CSRD data, we retrieve content based on similarity search. Given a question, part of a text, or simple keywords, we can access specific facts, articles and paragraph that we can trace back to our CSRD directive. This will become the foundation to our RAG strategy, answering specific regulatory compliance questions by citing existing chapters and articles. The example below returns the best matching paragraph with a relevance score given a user question.

# COMMAND ----------

from scripts.html_output import *

question = '''disclosing transactions between related parties, 
transactions between related parties included in a consolidation 
that are eliminated on consolidation shall not be included'''

search, score = CSRD_search.similarity_search_with_relevance_scores(
  question, 
  filters={"group": "PARAGRAPH"}
)[0]

displayHTML(vector_html(search.metadata['label'], search.page_content, '{}%'.format(int(score * 100))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundation model
# MAGIC It is expected that foundational models like DBRX, Llama 3, Mixtral or OpenAI that learned context from trillion of tokens (billions of documents) may already have acquired some knowledge for CSRD related questions. However, it would be cavalier to ask specific questions to a foundational model and solely rely on its built-in knowledge. In the context of regulatory compliance, we cannot afford for a model to return information as "best guess", regardless of how convincing its answer might be. 

# COMMAND ----------

# MAGIC %md
# MAGIC For the purpose of this exercise, we access DBRX model (provided as a service on your databricks workspace) through the langchain framework. Please note that most common external models (or fine tuned internal application) could be similarly served from the common interface of our [endpoint](https://docs.databricks.com/en/generative-ai/external-models/index.html) capability.

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 300, temperature=0)

# COMMAND ----------

query = 'Which disclosures will be subject to assurance, and what level of assurance is required?'
displayHTML(llm_html(query, chat_model.invoke(query).content))

# COMMAND ----------

# MAGIC %md
# MAGIC The question above might yield a well formed answer that, right or wrong, is definitely convincing to the naked eye. It may, however, lack substance and sound quality checks required for use in the domain of regulatory compliance. It would certainly be more comfortable for compliance officer to get actual facts and cross references to existing articles / paragraphs from a trusted source (such as the directive itself). Enters our RAG strategy.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## RAG strategy
# MAGIC Instead of solely relying on a model built-in knowlegde, we prompt a model to search for specific knowledge that we acquired throughout the first part of our notebook. This creates our foundation for RAG. We design a simple prompt and "chain" our vector store logic with actual model inference.

# COMMAND ----------

vector_retriever = load_vector_store_as_retriever()

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
    retriever=vector_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents = True
)

# COMMAND ----------

# MAGIC %md
# MAGIC Prompts must be designed in a way that provides user with better clarity and / or confidence as well as safeguarding model against malicious or offensive use. Outside of the scope for this solution, we encourage users to explore our [DB Demo](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot) that covers the most basics to the most advanced use of RAG and prompt engineering. A good example would be to restrict our model to only answer questions that are CSRD related, possibly linking multiple chains together. For further documentation, please refer to Databricks [Guardrails](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks) initiative.

# COMMAND ----------

question = {"query": 'Which disclosures are subject to assurance, and what level of assurance is required?'}
answer = chain.invoke(question)
displayHTML(rag_html(question['query'], answer['result'], answer['source_documents']))

# COMMAND ----------

# MAGIC %md
# MAGIC Through this simple example, we have guided our model to formulate a point of view solely based on facts we provided upfront (through our vector store), facts that we know we can trust. For the purpose of that demo, we represented output as a form of an HTML notebook. In real life scenario, one should offer that capability as a chat interface outside of a notebook based environment. Here is a simple example done using the [Streamlit](https://streamlit.io/) framework.

# COMMAND ----------

# MAGIC %md
# MAGIC ![demo_chatbot](https://raw.githubusercontent.com/databricks-industry-solutions/csrd_assistant/main/images/demo_chatbot.gif)
