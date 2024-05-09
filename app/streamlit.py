# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ['DATABRICKS_HOST'] = context.apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = context.apiToken().get()

# COMMAND ----------

from dbtunnel import dbtunnel

current_directory = os.getcwd()
script_path = current_directory + "/app.py"
dbtunnel.streamlit(script_path, port=8484).run()

# COMMAND ----------


