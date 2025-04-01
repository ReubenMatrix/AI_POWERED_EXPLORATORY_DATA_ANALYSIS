import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    @staticmethod
    def visualize_data(df):
        st.subheader("Data Visualization")

        viz_type = st.selectbox(
            "Select Visualization Type",
            [
                "Histogram",
                "Box Plot",
                "Correlation Heatmap"
            ]
        )

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns

        # Column selection
        selected_col = st.selectbox("Select Column", numeric_cols)

        # Create plot based on selection
        plt.figure(figsize=(10, 6))

        if viz_type == "Histogram":
            plt.hist(df[selected_col], bins=30)
            plt.title(f"Histogram of {selected_col}")
            plt.xlabel(selected_col)
            plt.ylabel("Frequency")

        elif viz_type == "Box Plot":
            sns.boxplot(x=df[selected_col])
            plt.title(f"Box Plot of {selected_col}")
            plt.xlabel(selected_col)

        elif viz_type == "Correlation Heatmap":
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0
            )
            plt.title("Correlation Heatmap")

        st.pyplot(plt)