<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-15.0ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/15.0ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?style=for-the-badge)](https://databricks.com/try-databricks)
[![POC](https://img.shields.io/badge/POC-3_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

*The CSRD (Corporate Sustainability Reporting Directive) is a European Union initiative aimed at enhancing corporate accountability regarding sustainability matters. It mandates certain companies to disclose non-financial information (such as environmental, social, and governance factors) in their annual reports and other public disclosures. Although this initiative may pose a data quality challenge to many firms collecting (and reporting on) this information for the first time, the [directive](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02013L0034-20240109&qid=1712714544806) itself (as per many regulatory documents) may be source of confusion / concerns and subject to various interpretation and therefore a perfect playground for generative AI application. In this solution accelerator co-developped with Deloitte France, we demonstrate how generative AI, retrieval augmented generation (RAG) and multi stage reasoning can be used to better navigate through the complexities of regulatory filings, bringing more transparency for companies to disclose their societal and environmental impacts.*

## Reference Architecture
![reference_architecture.png](https://raw.githubusercontent.com/databricks-industry-solutions/csrd_assistant/main/images/reference_architecture.png)

## Authors
<antoine.amend@databricks.com><br>
<corey.abshire@databricks.com>

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 

## License

&copy; 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
|beautifulsoup4|Parsing HTML|MIT|https://pypi.org/project/beautifulsoup4/|
|networkx|Graph library|BSD|https://pypi.org/project/networkx/|
|pyvis|Graph visualization|BSD|https://pypi.org/project/pyvis/|
|langchain|LLM framework|MIT|https://pypi.org/project/langchain/|
|langchain-openai|LLM framework for openAI|MIT|https://pypi.org/project/langchain-openai/|
|pydantic|Python hints|MIT|https://pypi.org/project/pydantic/|
|SQLAlchemy|Database abstraction|MIT|https://pypi.org/project/SQLAlchemy/|