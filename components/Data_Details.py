import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats


class DataDetails:
    @staticmethod
    def calculate_advanced_statistics(df):
        advanced_stats = {}

        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            column_data = df[column]

            advanced_stats[column] = {
                'Skewness': stats.skew(column_data),
                'Kurtosis': stats.kurtosis(column_data),
                'Median': np.median(column_data),
                'Range': np.ptp(column_data),
                'Variance': np.var(column_data),
                'Coefficient of Variation (%)': (np.std(column_data) / np.mean(column_data)) * 100 if np.mean(
                    column_data) != 0 else 0,
            }

        return pd.DataFrame.from_dict(advanced_stats, orient='index')

    @staticmethod
    def show_details(df):
        # Dataset Shape
        st.subheader("Dataset Overview")
        st.write("Dataset Shape:", df.shape)
        st.write("Total Rows:", df.shape[0])
        st.write("Total Columns:", df.shape[1])

        memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
        st.write(f"Memory Usage: {memory_usage:.2f} MB")

        st.subheader("Columns")
        columns_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Unique Values': df.nunique()
        })
        st.dataframe(columns_df)

        st.subheader("Basic Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("Advanced Statistical Measures")
        advanced_stats = DataDetails.calculate_advanced_statistics(df)
        st.dataframe(advanced_stats)