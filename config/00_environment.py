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
# MAGIC suppress warning

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

# MAGIC %md
# MAGIC Ensure paths

# COMMAND ----------

faiss_output_dir = '/Volumes/{}/{}/{}/faiss'.format(config['catalog'], config['schema'], config['volume'])
