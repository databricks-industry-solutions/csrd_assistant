# Databricks notebook source
# MAGIC %pip install -r ./requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import configparser
config = configparser.ConfigParser()
config.read('config/environment.ini')
config = config['DEMO']

# COMMAND ----------

# MAGIC %md
# MAGIC Suppress warning

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC Ensure catalog availability

# COMMAND ----------

def catalogs():
  return set(sql("SHOW CATALOGS").toPandas()['catalog'].to_list())

def schemas(catalog):
  return set(sql(f"SHOW DATABASES IN {catalog}").toPandas()['databaseName'].to_list())

def tables(catalog, schema):
  return set(sql(f"SHOW TABLES IN {catalog}.{schema}").toPandas()['tableName'].to_list())

def volumes(catalog, schema):
  return set(sql(f"SHOW VOLUMES IN {catalog}.{schema}").toPandas()['volume_name'].to_list())

# COMMAND ----------

if config['catalog'] not in catalogs():
  msg = 'Catalog [{}] does not exist'.format(config['catalog'])
  print(msg)
  try:
    _ = sql('CREATE CATALOG {}'.format(config['catalog']))
  except:
    raise Exception('Catalog creation failed')

if config['schema'] not in schemas(config['catalog']):
  msg = 'Schema [{}.{}] does not exist'.format(config['catalog'], config['schema'])
  print(msg)
  try:
    _ = sql('CREATE DATABASE {}.{}'.format(config['catalog'], config['schema']))
  except:
    raise Exception('Schema creation failed')

if config['volume'] not in volumes(config['catalog'], config['schema']):
  msg = 'Volume [/Volumes/{}/{}/{}] does not exist'.format(config['catalog'], config['schema'], config['volume'])
  print(msg)
  try:
    _ = sql('CREATE VOLUME {}.{}.{}'.format(config['catalog'], config['schema'], config['volume']))
  except:
    raise Exception('Schema creation failed')

# COMMAND ----------

# MAGIC %md
# MAGIC Ensure tables

# COMMAND ----------

table_nodes = '{}.{}.{}'.format(config['catalog'], config['schema'], config['table_nodes'])
table_edges = '{}.{}.{}'.format(config['catalog'], config['schema'], config['table_edges'])

# COMMAND ----------

_ = sql('''CREATE TABLE IF NOT EXISTS {db_table} (   
`id`        STRING COMMENT 'The identifier of text content, as a form of chapter.article.paragraph',
`label`     STRING COMMENT 'The business name of the text content, as referenced throughout document',
`content`   STRING COMMENT 'The actual text content for the given article, chapter or paragraph',
`group`     STRING COMMENT 'The type of content. Can be DIRECTIVE, CHAPTER, ARTICLE or PARAGRAPH'
) USING DELTA
TBLPROPERTIES(delta.enableChangeDataFeed = true)
COMMENT 'This table contains the actual text content from the CSRD directive. This content is synchronized to a vector store table that can be used for RAG application'
'''.format(db_table=table_nodes))

# COMMAND ----------

_ = sql('''CREATE TABLE IF NOT EXISTS {db_table} (   
`src`     STRING COMMENT 'The identifier of source text content, as a form of chapter.article.paragraph',
`dst`     STRING COMMENT 'The identifier of target text content, as a form of chapter.article.paragraph',
`label`   STRING COMMENT 'The relationship that exist between 2 pieces of content'
) USING DELTA
COMMENT 'This table contains the relationship between text content found in the CSRD directive. This table can be joined with nodes to access the underlying graph of the CSRD directive'
'''.format(db_table=table_edges))

# COMMAND ----------

# MAGIC %md
# MAGIC Ensure paths

# COMMAND ----------

faiss_output_dir = '/Volumes/{}/{}/{}/faiss'.format(config['catalog'], config['schema'], config['volume'])

# COMMAND ----------

# MAGIC %md
# MAGIC Ensure vector endpoints

# COMMAND ----------

vector_store_index_name = config['vector_index']
vector_store_index_coord = '{}.{}.{}'.format(config['catalog'], config['schema'], vector_store_index_name)
vector_store_endpoint_name = config['vector_endpoint']

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

def get_or_create_vs_endpoint():
  
  if vector_store_endpoint_name in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
    return vsc.get_endpoint(vector_store_endpoint_name)
    
  # Create vector index
  print(f"Creating endpoint [{vector_store_endpoint_name}], this can take a few min")
  vsc.create_endpoint(name=vector_store_endpoint_name, endpoint_type="STANDARD")  

  # Wait for provisioning
  for i in range(180):
    endpoint = vsc.get_endpoint(vector_store_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint [{vector_store_endpoint_name}]")
      time.sleep(10)
    else:
      raise Exception(f'''Error creating endpoint [{vector_store_endpoint_name}]''')
  raise Exception(f"Timeout, endpoint [{vector_store_endpoint_name}] isn't ready yet")

# COMMAND ----------

def wait_for_index(vs_index):
  for i in range(180):
    idx = vsc.get_index(vector_store_endpoint_name, vs_index).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return vsc.get_index(vector_store_endpoint_name, vs_index)
    elif "PROVISIONING" in status:
      if i % 20 == 0: 
        print(f"Index [{vector_store_endpoint_name}/{vs_index}] pipeline url:{url}")
        print(f"Waiting for index [{vector_store_endpoint_name}/{vs_index}]")
      time.sleep(10)
    else:
        raise Exception(f"Error creating index [{vector_store_endpoint_name}/{vs_index}]")
  raise Exception(f"Timeout when creating index [{vector_store_endpoint_name}/{vs_index}]")

# COMMAND ----------

def get_or_create_vs_index():
  
  indices = vsc.list_indexes(vector_store_endpoint_name).get("vector_indexes", list())
  indices = [index['name'] for index in indices]
  if vector_store_index_coord in indices:
    return vsc.get_index(vector_store_endpoint_name, vector_store_index_coord)

  # Create a new index
  print(f'Creating new vector index {vector_store_endpoint_name}/{vector_store_index_coord}, this can take a few minutes')
  vsc.create_delta_sync_index(
    endpoint_name=vector_store_endpoint_name,
    index_name=vector_store_index_coord,
    source_table_name=table_nodes,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column='content', 
    embedding_model_endpoint_name='databricks-bge-large-en'
  )

  # Wait for provisioning
  wait_for_index(vector_store_index_coord)

# COMMAND ----------

_ = get_or_create_vs_endpoint()
_ = get_or_create_vs_index()

# COMMAND ----------

def synchronize_vector_index():
  
  # synchronize index with table
  index = get_or_create_vs_index()
  index.sync()

  # wait for index to be ready
  status = 'N/A'
  i = 0
  while status != 'ONLINE_NO_PENDING_UPDATE':
    if i > 180:
      raise Exception('Could not synchronize index')
    idx = vsc.get_index(vector_store_endpoint_name, vector_store_index_coord).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    print('Waiting for index to be ready... [{}]'.format(status))
    time.sleep(10)
    i = i + 1

  return index

# COMMAND ----------

# MAGIC %md
# MAGIC Unfortunately, langchain does not seem to support query filters we can pass to our underlying vector store. Since we want to only retrieve paragraph content, we may need to override its business logic through a wrapper class, hardcoding our filter criteria.

# COMMAND ----------

from langchain_core.vectorstores import VectorStoreRetriever
from typing import Dict, List
from langchain.docstore.document import Document
from langchain_community.vectorstores import DatabricksVectorSearch

class VectorStoreRetrieverFilter(VectorStoreRetriever):
    def _get_relevant_documents(
        self,
        query: str
    ) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, filters={"group": "PARAGRAPH"}, **self.search_kwargs)
        return docs

# COMMAND ----------

def load_vector_store():
  return DatabricksVectorSearch(
    get_or_create_vs_index(), 
    text_column="content", 
    columns=["id", "label"]
  )

# COMMAND ----------

def load_vector_store_as_retriever():
  return VectorStoreRetrieverFilter(vectorstore=load_vector_store())
