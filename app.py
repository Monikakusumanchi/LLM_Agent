import os
import streamlit as st
from dotenv import load_dotenv
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

# Load environment variables
load_dotenv()

# Initialize database
db = SQLDatabase.from_uri('sqlite:///sql_lite_database.db')

# Initialize LLM with API key from environment variables
llm = OpenAI(
    temperature=0,
    verbose=True,
    openai_api_key= "",
)

# Setup the agent toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# Streamlit app interface
st.title("LLM Agent Bot")
user_question = st.text_input("Ask a question about the database:")

if user_question:
    answer = agent_executor.invoke(user_question)
    if answer:
        st.success(f"Answer: {answer}")
    else:
        st.warning("Sorry, I couldn't find an answer to your question in the database.")
