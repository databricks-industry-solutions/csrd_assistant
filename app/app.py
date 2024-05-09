import streamlit as st
import time
import os
import configparser

from langchain_core.vectorstores import VectorStoreRetriever
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient
from langchain_community.chat_models import ChatDatabricks
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


class VectorStoreRetrieverFilter(VectorStoreRetriever):
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        docs = self.vectorstore.similarity_search(query, filters={"group": "PARAGRAPH"}, **self.search_kwargs)
        return docs


def load_model(endpoint, index):

    template = """Given the context information and the chat history.
    Answer compliance issue related to the CSRD directive only.
    If the question is not related to regulatory compliance, kindly decline to answer. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Keep the answer as concise as possible, always citing articles and chapters whenever applicable.
    Please do not repeat the answer and do not add any additional information. 
    Context: {context}

    Chat History: {chat_history}
    Follow up question: {question}
    Assistant:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=load_chat_model(),
        retriever=load_vector_store(endpoint, index),
        combine_docs_chain_kwargs={'prompt': prompt},
        memory=memory,
        return_source_documents=True
    )


def load_chat_model():
    return ChatDatabricks(
        workspace_url=os.environ['DATABRICKS_HOST'], 
        personal_access_token=os.environ['DATABRICKS_TOKEN'],
        endpoint="databricks-dbrx-instruct", 
        max_tokens=300, 
        temperature=0, 
        extra_body={"enable_safety_filter": True}
        )


def load_vector_store(endpoint, index):
    vsc = VectorSearchClient(
        workspace_url=os.environ['DATABRICKS_HOST'], 
        personal_access_token=os.environ['DATABRICKS_TOKEN'],
        disable_notice=True
        )
    
    vector_index = vsc.get_index(endpoint, index)
    vector_store = DatabricksVectorSearch(vector_index, text_column="content", columns=["id", "label"])
    return VectorStoreRetrieverFilter(vectorstore=vector_store)


def bootstrap():
    config = configparser.ConfigParser()
    config.read('../config/environment.ini')
    config = config['DEMO']
    vector_endpoint = config['vector_endpoint']
    vector_index = "{}.{}.{}".format(config['catalog'], config['schema'], config['vector_index'])
    return load_model(vector_endpoint, vector_index)


def response_generator(answer_str):
    for word in answer_str.split():
        yield word + " "
        time.sleep(0.01)


def parse_history(messages):
    history = []
    for i, m in enumerate(messages):
        if i == len(messages) - 1:
            break
        history.append([m, messages[i + 1]])
    return history


# ------------------
# CORE STREAMLIT APP
# ------------------

st.image('european_commision.svg', width=300)
st.caption('''The Corporate Sustainability Reporting Directive (CSRD) is a European Union initiative 
aimed at enhancing corporate accountability regarding sustainability matters. It mandates certain companies to 
disclose non-financial information (such as environmental, social, and governance factors) in their annual reports 
and other public disclosures. This assistant provides users with the ability to navigate the complexity of the CSRD directive, its numerous chapters, articles and paragraphs.''')

# Initialize chat history and load model if necessary
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.llm = bootstrap()

# Display chat messages from history
for message in st.session_state.messages:
    if message['role'] == 'assistant':
        with st.chat_message(message["role"], avatar='assistant.png'):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar='user.png'):
            st.markdown(message["content"])

# Assign the user's input to the question variable and checked if it's not None in the same line
if question := st.chat_input("Ask me anything about the CSRD directive"):

    # Display user message in chat message container
    with st.chat_message("user", avatar='user.png'):
        st.markdown(question)

    # Define our model input
    model_input = {'question': question}

    # Query our model
    model_output = st.session_state.llm.invoke(model_input)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # Display response in chat message container
    with st.chat_message("assistant", avatar='assistant.png'):
        response = st.write_stream(response_generator(model_output['answer']))
        source_documents = [[doc.metadata['label'], doc.page_content] for doc in model_output['source_documents']]
        for article_id, article_content in source_documents:
            st.markdown(f":gray[**{article_id}**]")
            pre_text = f"""
                <body style="background: lightgray;">
                    <pre style="white-space: break-spaces;color:#9aa2b1;font-size: 0.6em">{article_content}</pre>
                </body>
            """
            st.html(pre_text)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

