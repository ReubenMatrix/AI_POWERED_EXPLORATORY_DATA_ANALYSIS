import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate


class DataReportGenerator:
    @staticmethod
    def sanitize_text(text):
        # Remove special characters while preserving useful punctuation
        text = re.sub(r'[^\w\s.,:]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Capitalize first letter of each sentence
        text = '. '.join(sentence.capitalize() for sentence in text.split('. '))

        return text

    @staticmethod
    def generate_ai_insights(df, api_key):
        try:
            # Set API Key
            os.environ["MISTRAL_API_KEY"] = api_key

            # Initialize Mistral AI model
            model = ChatMistralAI(model="mistral-large-latest")

            # Prepare comprehensive context for AI analysis
            context = f"""
            Detailed Dataset Characteristics:
            - Total Rows: {df.shape[0]}
            - Total Columns: {df.shape[1]}
            - Columns: {', '.join(df.columns)}

            Column Types and Unique Values:
            {df.dtypes}
            Unique Values: {df.nunique()}

            Numeric Columns Statistics:
            {df.describe().T.to_string()}

            Missing Values:
            {df.isnull().sum()}
            """

            # Create a comprehensive prompt template with improved structure
            prompt_template = PromptTemplate(
                input_variables=["context"],
                template="""Provide a comprehensive, structured analysis of the dataset using a professional tone:

                Dataset Context:
                {context}

                Analysis Guidelines:
                1. Provide insights in a clear, structured format
                2. Use professional language and avoid casual expressions
                3. Separate different types of insights with clear headings
                4. Ensure each insight is concise and actionable
                5. Include potential challenges and recommendations
                6. Keep each point as a single sentence

                Desired Output Format:
                ## Key Observations
                - Insight 1
                - Insight 2

                ## Data Quality
                - Observation 1
                - Observation 2

                ## Preprocessing Recommendations
                - Recommendation 1
                - Recommendation 2

                ## Potential Analysis Approaches
                - Approach 1
                - Approach 2

                ## Machine Learning Potential
                - Linear regression: brief description
                - Decision trees/random forests: brief description
                - Gradient boosting machines: brief description
                - Support vector machines (SVM): brief description

                ## Drawbacks 
                - Drawback 1
                - Drawback 2

                Ensure each point is a simple, single sentence that will be displayed on its own line.
                Do not combine multiple insights into one bullet point.
                """
            )

            # Generate insights
            prompt = prompt_template.format(context=context)
            insights = model.invoke(prompt).content

            # Process and format insights
            def format_insights(text):
                # Split insights into sections by section headers (##)
                sections = re.split(r'(##\s*[^\n]+)', text)
                formatted_sections = []

                current_section = None
                current_points = []

                for i, section in enumerate(sections):
                    section = section.strip()

                    # Skip empty sections
                    if not section or len(section) < 3:
                        continue

                    if section.startswith('##'):
                        # If we have a previous section, add it to the results
                        if current_section and current_points:
                            formatted_sections.append({
                                'heading': current_section,
                                'points': current_points
                            })

                        # Start a new section
                        current_section = section.replace('##', '').strip()
                        current_points = []
                    else:
                        # Extract bullet points
                        bullet_points = re.split(r'\n\s*-\s*|\n\s*•\s*', section)

                        for point in bullet_points:
                            if not point.strip():
                                continue

                            # Split individual sentences
                            sentences = re.split(r'(?<=[.!?])\s+', point.strip())

                            for sentence in sentences:
                                if sentence.strip():
                                    current_points.append(DataReportGenerator.sanitize_text(sentence.strip()))

                # Don't forget the last section
                if current_section and current_points:
                    formatted_sections.append({
                        'heading': current_section,
                        'points': current_points
                    })

                return formatted_sections

            # Process and return formatted insights
            return format_insights(insights)

        except Exception as e:
            return [{
                'heading': 'Error',
                'points': [f"Error generating AI insights: {str(e)}"]
            }]

    @staticmethod
    def create_data_table(data, title=None):
        """
        Create a formatted table for the PDF report
        """
        if isinstance(data, pd.DataFrame):
            table_data = [data.columns.tolist()] + data.values.tolist()
        elif isinstance(data, dict):
            table_data = [list(data.keys()), list(data.values())]
        else:
            return None

        from reportlab.platypus import Table, TableStyle
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table

    @staticmethod
    def generate_dataset_report(df, api_key=None):
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)

        # Create custom styles for better formatting
        styles = getSampleStyleSheet()

        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=20,
            spaceBefore=10,
        )

        heading1_style = ParagraphStyle(
            'Heading1',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=TA_LEFT,
            spaceBefore=15,
            spaceAfter=10,
            fontName='Helvetica-Bold',
        )

        heading2_style = ParagraphStyle(
            'Heading2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=8,
            fontName='Helvetica-Bold',
        )

        # Create a new highlighted subheading style
        subheading_style = ParagraphStyle(
            'Subheading',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            textColor=colors.darkblue,
            borderWidth=0.5,
            borderColor=colors.lightgrey,
            borderPadding=5,
            backColor=colors.lightgrey,
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceBefore=6,
            spaceAfter=6,
        )

        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=10,
            spaceBefore=3,
            spaceAfter=3,
            leftIndent=20,
        )

        story = []

        # Title
        story.append(Paragraph("Comprehensive Dataset Analysis Report", title_style))
        story.append(Spacer(1, 18))

        # Dataset Overview
        story.append(Paragraph("1. Dataset Overview", heading1_style))
        story.append(Spacer(1, 8))

        overview_data = {
            'Total Rows': df.shape[0],
            'Total Columns': df.shape[1],
            'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB"
        }
        overview_table = DataReportGenerator.create_data_table(overview_data)
        story.append(overview_table)
        story.append(Spacer(1, 15))

        # Columns Details
        story.append(Paragraph("2. Columns Details", heading1_style))
        story.append(Spacer(1, 8))

        columns_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Unique Values': df.nunique()
        })
        columns_table = DataReportGenerator.create_data_table(columns_df)
        story.append(columns_table)
        story.append(Spacer(1, 15))

        # Statistical Summary
        story.append(Paragraph("3. Statistical Summary", heading1_style))
        story.append(Spacer(1, 8))

        stats_df = df.describe().T
        stats_table = DataReportGenerator.create_data_table(stats_df)
        story.append(stats_table)
        story.append(Spacer(1, 15))

        # AI-Powered Insights
        if api_key:
            story.append(Paragraph("4. AI-Powered Insights", heading1_style))
            story.append(Spacer(1, 8))

            try:
                ai_insights_sections = DataReportGenerator.generate_ai_insights(df, api_key)

                # Process each section with its own heading and bullet points
                for section in ai_insights_sections:
                    # Add the section heading with highlighted style
                    story.append(Paragraph(section['heading'], subheading_style))
                    story.append(Spacer(1, 5))

                    # Add each point as a bullet point
                    for point in section['points']:
                        story.append(Paragraph(f"• {point}", bullet_style))
                        story.append(Spacer(1, 3))

                    # Add space after each section
                    story.append(Spacer(1, 10))

                # Add extra space after all insights
                story.append(Spacer(1, 15))

            except Exception as e:
                story.append(Paragraph(f"Error generating AI insights: {str(e)}", normal_style))
                story.append(Spacer(1, 15))

        # Correlation Heatmap
        story.append(Paragraph("5. Correlation Analysis", heading1_style))
        story.append(Spacer(1, 8))

        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()

            # Set up the heatmap with improved aesthetics
            heatmap = sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                linewidths=0.5,
                annot_kws={"size": 8}
            )
            plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
            plt.tight_layout()

            correlation_buffer = io.BytesIO()
            plt.savefig(correlation_buffer, format='png', bbox_inches='tight', dpi=150)
            correlation_buffer.seek(0)
            correlation_img = Image(correlation_buffer, width=6 * inch, height=4 * inch)
            story.append(correlation_img)
            plt.close()
        else:
            story.append(Paragraph("Insufficient numeric columns for correlation analysis", normal_style))

        # Build PDF with improved formatting
        doc.build(story)

        # Return PDF buffer
        buffer.seek(0)
        return buffer

    @staticmethod
    def display_download_button(df):
        st.subheader("Generate Comprehensive Dataset Report")

        # API Key input (consider moving to a more secure method)
        api_key = ''

        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF Report..."):
                try:
                    pdf_buffer = DataReportGenerator.generate_dataset_report(df, api_key)

                    st.download_button(
                        label="Download Dataset Report",
                        data=pdf_buffer,
                        file_name="dataset_analysis_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")