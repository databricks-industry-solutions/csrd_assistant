<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-15.0ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/15.0ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?style=for-the-badge)](https://databricks.com/try-databricks)
[![POC](https://img.shields.io/badge/POC-3_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

*Organizations are rushing to comply with new CSRD (Corporate Sustainability Reporting Directive) regulations before the January 1st, 2025 deadline. The extensive 200+ page legal document outlining the CSRD regulation and ESRS reporting standards has been revised multiple times over the past decade and now applies to a broader range of organizations. Deloitte and Databricks have joined forces to design and build an accelerator that combines the best of both worlds: the expertise of a leading professional services consulting firm with the cutting-edge technology of a top data vendor. This collaboration provides a comprehensive solution integrating the latest Data + AI technologies, including GenAI, LLM, prompt engineering, vector search, Langchain, and more. These tools are designed to be easily understood and utilized by business professionals using natural language.*

## Scope

Our vision is to empower organizations by provisioning 2 key capabilities:
Ensure that individuals working on sustainability reports are thoroughly familiar with the CSRD rules, enabling them to perform their duties without risk of infringement. This complementary tool will seamlessly integrate with the ESRS reporting standards and workflows that organizations must establish to meet CSRD reporting requirements.
Serve as a catalyst for aligning with upcoming industry-specific variations of the ESRS reporting standards, such as those for Oil and Gas, Motor Vehicles, and Energy production utilities. This will facilitate a quicker and more thorough understanding of the different reporting requirements and associated indicators.
 
Until the sectorial ESRS reporting standard are issued, the solution could also integrate directives issued by sector specific regulators (e.g. ACPR for financial services in France) related to sustainability reporting and help organizations adapt their CSRD reporting to their sectors.
 
Broadly speaking, the accelerator could also be applied to other regulatory texts that require specific reporting, such as the AI Act. This ensures that when new regulations are enacted, organizations will have reliable methodology to address complex regulatory requirements swiftly and efficiently.

## Reference Architecture
![reference_architecture.png](https://raw.githubusercontent.com/databricks-industry-solutions/csrd_assistant/main/images/reference_architecture.png)

## Authors
<antoine.amend@databricks.com><br>
<corey.abshire@databricks.com><br>
<elarroze@deloitte.fr>


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