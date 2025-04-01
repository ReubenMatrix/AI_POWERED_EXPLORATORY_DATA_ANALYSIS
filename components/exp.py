import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, inspect, text
from smolagents import tool, CodeAgent, HfApiModel


class NLPChat:
    def __init__(self):
        # Database configuration
        self.USER = "root"
        self.PASSWORD = "root"
        self.HOST = "localhost"
        self.DATABASE = "mydb"
        self.DB_URI = f"mysql+pymysql://{self.USER}:{self.PASSWORD}@{self.HOST}/{self.DATABASE}"
        self.engine = None

        # Hardcoded Hugging Face API key (Not Recommended)


    def establish_connection(self):
        try:
            # Create SQLAlchemy engine
            self.engine = create_engine(self.DB_URI)
            with self.engine.connect() as connection:
                st.success("Database connection established successfully!")
                return self.engine
        except Exception as e:
            st.error(f"Error connecting to database: {e}")
            return None

    def upload_csv(self):
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            self.upload_to_database(df, uploaded_file.name)

    def upload_to_database(self, df, filename):
        engine = self.establish_connection()
        if engine:
            try:
                table_name = filename.replace('.csv', '').lower()
                df.to_sql(table_name, engine, if_exists='replace', index=False)
                st.success(f"CSV file uploaded to table '{table_name}' successfully!")
                self.setup_sql_tool(table_name)
            except Exception as e:
                st.error(f"Error uploading to database: {e}")

    def setup_sql_tool(self, table_name):
        @tool
        def sql_engine(query: str) -> str:
            """
                Execute SQL queries on the uploaded table.

                Args:
                    query (str): The SQL query to execute.

                Returns:
                    str: The output of the query execution.
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

        # Initialize CodeAgent with the hardcoded API key
        HF_API_KEY = "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"
        agent = CodeAgent(
            tools=[sql_engine],

            model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
        )

        st.subheader("SQL Query Interface")
        query = st.text_input("Enter SQL Query", key="sql_query")
        if st.button("Execute Query"):
            try:
                result = agent.run(query)
                st.write("Query Result:")
                st.code(result)
            except Exception as e:
                st.error(f"Error: {e}")


def main():
    st.title("NLP CSV Uploader and SQL Query Tool")
    nlp_chat = NLPChat()
    nlp_chat.upload_csv()


if __name__ == "__main__":
    main()
