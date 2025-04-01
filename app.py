import streamlit as st
from components.Data_Loader import DataLoader
from components.Data_Visualization import DataVisualizer
from components.Data_Details import DataDetails
from components.Data_Report import DataReportGenerator
from components.Data_Chat_NLP import NLPChat


def main():
    st.title("AI-Powered EDA Agent")
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = DataLoader.load_csv(uploaded_file)
        duplicate_count = df.duplicated().sum()

        print(f"Number of duplicate rows: {duplicate_count}")

        if df is not None:
            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "Dataset Details",
                "Data Visualization",
                "Chat with dataset",
            ])

            nlp_chat = NLPChat()

            with tab1:

                DataDetails.show_details(df)
                DataReportGenerator.display_download_button(df)

                try:
                    nlp_chat.upload_to_database(df)
                except Exception as e:
                    st.error(f"Database insertion error: {e}")

            with tab2:
                DataVisualizer.visualize_data(df)

            with tab3:
                nlp_chat.display(df)


if __name__ == "__main__":
    main()
