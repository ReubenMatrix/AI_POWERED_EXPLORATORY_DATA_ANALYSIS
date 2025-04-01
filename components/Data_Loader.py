import pandas as pd
import streamlit as st


class DataLoader:
    @staticmethod
    def load_csv(uploaded_file):
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None