# Databricks notebook source
# MAGIC %md
# MAGIC ## Indexing content
# MAGIC Though representing our CSRD directive as a graph was visually compelling, it offers no semantic search capability. In this section, we will further index our graph data to offer search functionality for a given question / query that can be combined with Generative AI capabilities as part of a Retrieval Augmented Generation strategy (RAG). 
# MAGIC
# MAGIC In production settings, we highly encourage users to leverage [Databricks vector store](https://docs.databricks.com/en/generative-ai/vector-search.html) capability, linking records to the actual binary file that may have been previously stored on your volume, hence part of a unified governance strategy. 
# MAGIC
# MAGIC However, in the context of a solution accelerator, we decided to limit the infrastructure requirements (volume creation, table creation, DLT pipelines, etc.) and provide entry level capabilities, in memory, leveraging FAISS as our de facto vector store and leveraging out-of-the-box foundation models provided by Databricks.
# MAGIC
# MAGIC For more information on E2E applications, please refer to [DB Demo](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot).

# COMMAND ----------

# MAGIC %run ./config/00_environment

# COMMAND ----------

nodes_df = spark.read.table(table_nodes).toPandas()
display(nodes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We leverage the langchain framework for its ease of use, converting our graph content as a set of langchain documents that we can index on our vector store. Please note that chunking is no longer required since we have been able to extract granular information (at a paragraph level) in our previous notebook.

# COMMAND ----------


from langchain.docstore.document import Document

documents = []
for i, n in nodes_df[nodes_df['group'] == 'PARAGRAPH'].iterrows():
  metadata=n.to_dict()
  page_content=metadata['content']
  del metadata['content']
  documents.append(Document(
    page_content=page_content,
    metadata=metadata
  ))


# COMMAND ----------

# MAGIC %md
# MAGIC For our embedding strategy, we will leverage foundational models provided out of the box through your development workspace environment.

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

from langchain.vectorstores import FAISS
CSRD_search = FAISS.from_documents(documents=documents, embedding=embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now retrieve content based on similarity search. Given a question, part of a text, or simple keywords, we retrieve specific facts, articles and paragraph that we can trace back to our CSRD directive. This will become the foundation to our RAG strategy later, answering specific regulatory compliance questions by citing existing chapters and articles. The example below returns the best matching paragraph with a relevance score given a user question.

# COMMAND ----------

from scripts.html_output import *

question = '''disclosing transactions between related parties, 
transactions between related parties included in a consolidation 
that are eliminated on consolidation shall not be included'''

search, score = CSRD_search.similarity_search_with_relevance_scores(question)[0]
displayHTML(vector_html(search.metadata['label'], search.page_content, '{}%'.format(int(score * 100))))

# COMMAND ----------

# MAGIC %md
# MAGIC We serialize our vector store for future use. Once again, we highly recommend leveraging vector store capability instead of local FAISS used for demo purpose only. 

# COMMAND ----------

CSRD_search.save_local(faiss_output_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundation model
# MAGIC It is expected that foundational models like DBRX, Llama 3, Mixtral or OpenAI that learned knowledge from trillion of tokens may already have acquired some knowledge for generic questions related to the CSRD directive. However, it would be cavalier to ask specific questions to our foundation model and solely rely on its built-in knowledge we could not link to exact articles. In the context of regulatory compliance, we cannot afford for a model to return information as "best guess", regardless of how convincing its answer might be. We leverage DBRX model (provided as a service on your databricks workspace).

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 300, temperature=0)

# COMMAND ----------

query = 'Which disclosures will be subject to assurance, and what level of assurance is required?'
displayHTML(llm_html(query, chat_model.invoke(query).content))

# COMMAND ----------

# MAGIC %md
# MAGIC The question above might yield a well formed answer that may seem convincing to the naked eye, but definitely lacks substance and sound quality required for regulatory compliance. Instead of an output we may have to take at its face value, it would certainly be more comfortable to cite actual facts and cross references to existing articles / paragraphs from a trusted source (such as the directive itself).

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
    retriever=CSRD_search.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents = True
)

# COMMAND ----------

# MAGIC %md
# MAGIC Prompts must be designed in a way that provides user with better clarity and / or confidence as well as safeguarding model against malicious or offensive use. This, however, is not part of this solution. We invite users to explore our [DB Demo](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot) that covers the basics to most advanced use of RAG and prompt engineering. A good example would be to restrict our model to only answer questions that are CSRD related, possibly linking multiple chains together.

# COMMAND ----------

question = {"query": 'Which disclosures will be subject to assurance, and what level of assurance is required?'}
answer = chain.invoke(question)
displayHTML(rag_html(question['query'], answer['result'], answer['source_documents']))

# COMMAND ----------

# MAGIC %md
# MAGIC In this simple example, we have let our model formulate a point of view based on actual facts we know we can trust. For the purpose of that demo, we represented output as a form of an HTML notebook. In real life scenario, we should offer that capability as a chat interface to a set of non technologist user, hence requiring building a UI and application server outside of a notebook based environment (outside of the scope here). 
