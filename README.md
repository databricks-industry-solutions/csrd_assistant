<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![DBR](https://img.shields.io/badge/15.0ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/CHANGE_ME.html)
[![CLOUD](https://img.shields.io/badge/All-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
[![EFFORT](https://img.shields.io/badge/2_days-orange?style=for-the-badge)](https://databricks.com/try-databricks)


## CSRD assistant
*On July 31, 2023, the European Commission adopted the [European Sustainability Reporting Standards](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202302772) (ESRS), which were published in the Official Journal of the European Union in December 2023. Drafted by the European Financial Reporting Advisory Group (EFRAG), the standards provide supplementary guidance for companies within the scope of the [E.U. Corporate Sustainability Reporting Directive](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022L2464) (CSRD). The adoption of the CSRD, along with the supporting ESRS, is intended to increase the breadth of nonfinancial information reported by companies and to ensure that the information reported is consistent, relevant, comparable, reliable, and easy to access. Source: [Deloitte](https://dart.deloitte.com/USDART/home/publications/deloitte/heads-up/2023/csrd-corporate-sustainability-reporting-directive-faqs)*

Though the CSRD compliance poses a data quality challenge to firms trying to collect and report this information for the first time, the directive itself (as per many regulatory documents) may be source of confusion / concerns and subject to interpretation. In this exercise, we want to demonstrate generative AI abilities to navigate through the complexities of regulatory documentation. 

## Reference Architecture
![reference_architecture.png](https://raw.githubusercontent.com/databricks-industry-solutions/csrd_assistant/main/images/reference_architecture.png)

## Authors
<antoine.amend@databricks.com>

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
|faiss-cpu|In memory vector store|MIT|https://pypi.org/project/faiss-cpu/|
|langchain|LLM framework|MIT|https://pypi.org/project/langchain/|
|SQLAlchemy|Database abstraction|MIT|https://pypi.org/project/SQLAlchemy/|