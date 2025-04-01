import streamlit as st
import os
import pandas as pd
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from io import StringIO


class DataChat:
    @staticmethod
    def prepare_csv_context(df, max_rows=50):
        buffer = StringIO()
        df.info(buf=buffer)
        buffer.seek(0)
        df_info = buffer.getvalue()

        # Prepare data description
        data_description = f"""
        Dataset Overview:
        - Total Rows: {df.shape[0]}
        - Total Columns: {df.shape[1]}
        - Columns: {', '.join(df.columns)}

        Column Types:
        {df.dtypes}

        Summary Statistics:
        {df.describe().to_string()}

        Data Info:
        {df_info}

        Sample Data (first {min(max_rows, len(df))} rows):
        {df.head(max_rows).to_csv(index=False)}
        """
        return data_description

    @staticmethod
    def initialize_chat(df):
        # Prepare CSV context
        csv_context = DataChat.prepare_csv_context(df)

        api_key = 'R7dMbTb2uzWiPbDM6zqREQHe0PUCXb7f'

        # Initialize session state for chat
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat interface
        st.subheader("Chat with Your Data")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # API Key and Model Initialization
        if api_key:
            try:
                os.environ["MISTRAL_API_KEY"] = api_key
                model = ChatMistralAI(model="mistral-large-latest")

                # Create a comprehensive prompt template
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=f"""You are an advanced data analysis assistant specialized in CSV data exploration.
                    Your goal is to provide insightful, accurate, and helpful analysis of the uploaded dataset.

                    Dataset Context:
                    {csv_context}

                    Guidelines:
                    1. Use the provided dataset context to inform your responses
                    2. Be precise and data-driven in your analysis
                    3. If a specific analysis is not possible with the given data, explain why
                    4. Offer suggestions for further investigation
                    5. Provide clear, concise, and actionable insights

                    You have access to the full dataset and its metadata. 
                    Feel free to reference specific columns, statistics, or patterns you observe."""),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])

                # Create conversation chain
                conversation = ConversationChain(
                    llm=model,
                    prompt=prompt,
                    memory=st.session_state.memory,
                    verbose=False
                )

                # User input
                if user_question := st.chat_input("Ask a question about your data..."):
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_question})

                    with st.chat_message("user"):
                        st.markdown(user_question)

                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing data..."):
                            # Generate response
                            response = conversation.predict(input=user_question)
                            st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                    # Add download button for chat export
                    if st.session_state.chat_history:
                        chat_text = "\n\n".join(
                            [f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])

                        # Convert string to bytes for download
                        bytes_data = chat_text.encode()

                        st.download_button(
                            label="Download Chat",
                            data=bytes_data,
                            file_name="data_chat_export.txt",
                            mime="text/plain"
                        )

            except Exception as e:
                st.error(f"Error initializing Mistral AI: {str(e)}")
                st.warning("Please check your API key and try again.")
        else:
            st.info("Please enter your Mistral AI API key to start chatting with your data.")

        # Optional: Show dataset preview
        with st.expander("Dataset Preview"):
            st.dataframe(df.head())
