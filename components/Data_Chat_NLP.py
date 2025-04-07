import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from smolagents import tool, CodeAgent, HfApiModel
import re


class NLPChat:
    def __init__(self):
        self.USER = "root"
        self.PASSWORD = "root"
        self.HOST = "localhost"
        self.DATABASE = "mydb"
        self.DB_URI = f"mysql+pymysql://{self.USER}:{self.PASSWORD}@{self.HOST}/{self.DATABASE}"
        self.engine = None
        self.current_table_name = None

    def sanitize_table_name(self, name):
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        if not sanitized_name[0].isalpha():
            sanitized_name = 'table_' + sanitized_name
        return sanitized_name.lower()

    def establish_connection(self):
        try:
            if self.engine is None:  # Prevent multiple connections
                self.engine = create_engine(self.DB_URI)
            with self.engine.connect():
                st.success("Database connected successfully!")
                return self.engine
        except Exception as e:
            st.error(f"Error connecting to database: {e}")
            return None

    def upload_to_database(self, df, filename=None):
        engine = self.establish_connection()
        if engine:
            try:
                filename = filename or "uploaded_dataset"
                table_name = self.sanitize_table_name(filename)
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                self.current_table_name = table_name
                st.success(f"Dataset uploaded to table '{table_name}' successfully!")
            except Exception as e:
                st.error(f"Error uploading dataset: {e}")

    def setup_sql_tool(self):
        if not self.engine or not self.current_table_name:
            st.error("No database connection or table available!")
            return

        @tool
        def sql_engine(query: str) -> str:
            """Execute SQL queries on the uploaded table.

            Args:
                query: The SQL query to execute.

            Returns:
                The output of the query execution as a string.
            """
            try:
                output = ""
                with self.engine.connect() as con:
                    rows = con.execute(text(query))
                    for row in rows:
                        output += "\n" + str(row)
                return output
            except Exception as e:
                return f"Error executing query: {e}"


        # Initialize CodeAgent
        HF_API_KEY = ""
        agent = CodeAgent(
            tools=[sql_engine],
            model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
        )

        st.subheader("Chat with your dataset")
        query = st.text_input("Enter SQL Query", key="sql_query")
        if st.button("Execute Query"):
            try:
                result = agent.run(query)
                st.write("📜 Query Result:")
                st.code(result)
            except Exception as e:
                st.error(f"Error executing query")

    def display(self, df):

        if self.current_table_name is None:
            self.upload_to_database(df, df.name if hasattr(df, 'name') else 'uploaded_dataset')

        if self.establish_connection() and self.current_table_name:
            self.setup_sql_tool()

