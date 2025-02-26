from transformers import pipeline
from app import analyze_video_comments
import plotly.express as px
import streamlit as st
import pandas as pd
import random
import torch
import time
import html
import re

st.set_page_config(layout='wide')

st.markdown("# YouTube Comment Sentiment Analysis 🎥 ")
st.markdown("#### :gray[Easily discover the sentiment behind YouTube comments and understand what people are saying]")
st.markdown("#### :gray[Analyze the sentiment of YouTube video comments (Positive, Neutral, Negative)]")
st.markdown("---")

url = st.sidebar.text_input("Enter YouTube video URL:")

# Store session states for comments and actions
if 'positive_comments' not in st.session_state:
    st.session_state.positive_comments = []
if 'neutral_comments' not in st.session_state:
    st.session_state.neutral_comments = []
if 'negative_comments' not in st.session_state:
    st.session_state.negative_comments = []
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'default'  # Track which button is currently active

# Sidebar buttons for navigation
if st.sidebar.button("Get Sentiment"):
    if url:
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        if match:
            video_id = match.group(1)
            st.write("### Fetching and analyzing comments...")
            try:
                positive_comments, neutral_comments, negative_comments = analyze_video_comments(video_id)

                # Store results in session state
                st.session_state.positive_comments = positive_comments
                st.session_state.neutral_comments = neutral_comments
                st.session_state.negative_comments = negative_comments
                st.session_state.active_page = 'analysis'  # Switch to analysis page
            except Exception as e:
                st.error(f"Error fetching comments: {e}")
        else:
            st.sidebar.error("Invalid YouTube URL. Please enter a valid URL.")
    else:
        st.sidebar.error("Please enter a YouTube video URL.")


if st.sidebar.button("Generate Summary"):
    if st.session_state.positive_comments or st.session_state.neutral_comments or st.session_state.negative_comments:
        st.session_state.active_page = 'summary'  # Switch to summary page
    else:
        st.sidebar.error("No comments analyzed. Please use the 'Analyze' button first.")

if st.sidebar.button("Visualization"):
    if st.session_state.positive_comments or st.session_state.neutral_comments or st.session_state.negative_comments:
        st.session_state.active_page = 'visualization'  # Switch to visualization page
    else:
        st.sidebar.error("No comments analyzed. Please use the 'Analyze' button first.")

# ======================================================================================================================

# SENTIMENT PART
if st.session_state.active_page == 'analysis':
    st.write(f"### :blue[Sentiment Counts]")
    st.write(f"- Positive Comments: {len(st.session_state.positive_comments)}")
    st.write(f"- Neutral Comments: {len(st.session_state.neutral_comments)}")
    st.write(f"- Negative Comments: {len(st.session_state.negative_comments)}")
    st.markdown("---")

    sentiment_type = st.radio(
        "Choose sentiment to view comments:",
        options=['Positive', 'Neutral', 'Negative'],
        format_func=lambda x: f":green[{x}]" if x == 'Positive' else f":gray[{x}]" if x == 'Neutral' else f":red[{x}]"
    )
    st.markdown("\n")

    if sentiment_type == 'Positive' and st.session_state.positive_comments:
        st.write("## Positive Comments")
        for comment in random.sample(st.session_state.positive_comments, min(20, len(st.session_state.positive_comments))):
            st.write(f"- {html.unescape(comment)}")
    elif sentiment_type == 'Neutral' and st.session_state.neutral_comments:
        st.write("## Neutral Comments")
        for comment in random.sample(st.session_state.neutral_comments, min(20, len(st.session_state.neutral_comments))):
            st.write(f"- {html.unescape(comment)}")
    elif sentiment_type == 'Negative' and st.session_state.negative_comments:
        st.write("## Negative Comments")
        for comment in random.sample(st.session_state.negative_comments, min(20, len(st.session_state.negative_comments))):
            st.write(f"- {html.unescape(comment)}")
    else:
        st.write(f"No {sentiment_type.lower()} comments available.")

# ======================================================================================================================

# SUMMARY PART
elif st.session_state.active_page == 'summary':
    st.write("## Get your Summary")

    if st.session_state.positive_comments or st.session_state.neutral_comments or st.session_state.negative_comments:
        # summarizer = pipeline("summarization")  #old
        summarizer = pipeline("summarization", device=0 if torch.cuda.is_available() else -1)   #new

        # Radio button for sentiment selection
        sentiment_type = st.radio(
            "Choose sentiment to summarize:",
            options=['Positive', 'Neutral', 'Negative', 'All Comments'],
            format_func=lambda x: f":green[{x}]" if x == 'Positive' else f":gray[{x}]" if x == 'Neutral' else f":red[{x}]" if x == 'Negative' else f":blue[{x}]"
        )
        st.write('')

        # "Generate" button
        generate_summary = st.button("Generate")

        # Function to generate the summary word-by-word
        def generate_summary_stream(comments, sentiment_label):
            """Generates and displays the summary incrementally."""
            if comments:

                # Prepare input for summarization
                summary_input = " ".join(comment.strip() for comment in comments if comment.strip())
                if len(summary_input.split()) >= 30:  # Ensure enough words for summarization
                    try:
                        # Generate the summary text
                        summary = summarizer(summary_input, max_length=500, min_length=100, do_sample=False)
                        # summary = summarizer(summary_input, do_sample=False)
                        full_summary = summary[0]['summary_text']

                        # Stream the output word-by-word
                        st.write(f"### {sentiment_label} Summary:")
                        summary_placeholder = st.empty()
                        current_text = ""  # Initialize an empty string for the growing summary
                        for word in full_summary.split():
                            current_text += word + " "  # Append the next word with a space
                            summary_placeholder.text(current_text.strip())  # Update the placeholder
                            time.sleep(0.1)  # Small delay to simulate real-time streaming
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {e}")
                else:
                    st.error(f"The {sentiment_label} comments are too short for summarization.")
            else:
                st.write(f"No {sentiment_label.lower()} comments available.")

        # Only generate the summary if the button is clicked
        if generate_summary:
            if sentiment_type == 'Negative':
                generate_summary_stream(st.session_state.negative_comments[:30], "Negative")
            elif sentiment_type == 'Positive':
                generate_summary_stream(st.session_state.positive_comments[:30], "Positive")
            elif sentiment_type == 'Neutral':
                generate_summary_stream(st.session_state.neutral_comments[:30], "Neutral")
            elif sentiment_type == 'All Comments':
                # Take 10 from each category if available
                mixed_comments = (
                    st.session_state.positive_comments[:10]
                    + st.session_state.neutral_comments[:10]
                    + st.session_state.negative_comments[:10]
                )
                generate_summary_stream(mixed_comments, "Mixed")
    else:
        st.error("No comments available for summarization. Please analyze a video first.")

# ======================================================================================================================

# VISUALIZATION PART
elif st.session_state.active_page == 'visualization':
    st.write("## Visualization of Sentiment Analysis")

    COL1, COL2 = st.columns([3.5, 1.2])
    with COL1:
        positive_comments = len(st.session_state.positive_comments)
        neutral_comments = len(st.session_state.neutral_comments)
        negative_comments = len(st.session_state.negative_comments)
        total_comments = positive_comments + neutral_comments + negative_comments

        if total_comments > 0:
            positive_percentage = (positive_comments / total_comments) * 100
            neutral_percentage = (neutral_comments / total_comments) * 100
            negative_percentage = (negative_comments / total_comments) * 100
        else:
            positive_percentage = neutral_percentage = negative_percentage = 0

        data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Percentage': [positive_percentage, neutral_percentage, negative_percentage]})

        fig = px.bar(data, x='Percentage', y='Sentiment', text='Percentage', orientation='h',
                     color='Sentiment', color_discrete_map={'Positive': '#52ac18', 'Neutral': '#aab7a9', 'Negative': '#c51515'},
                     labels={'Percentage': 'Percentage (%)'}, title='Visualized your data')

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='inside')

        # Adjust layout to add grid and change bar width
        fig.update_layout(
            xaxis=dict(showgrid=True),  # Show grid on x-axis
            yaxis=dict(showgrid=True),  # Show grid on y-axis
            bargap=0.5,  # Thicker bars (reduce gap between bars)
            title_x=0.5)  # Center the title
        st.plotly_chart(fig)
    st.markdown('---')

    import cupy as cp
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import streamlit as st

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    colUM1, colUM2 = st.columns([3.5, 1.2])
    with colUM1:
        if st.session_state.active_page == 'visualization':
            st.write("")
            st.write("")
            all_comments = (
                    st.session_state.positive_comments
                    + st.session_state.neutral_comments
                    + st.session_state.negative_comments)
            mixed_comments_text = " ".join(all_comments)

            # Convert text into word frequencies (this step remains CPU-based)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=200).generate(mixed_comments_text)

            # Convert to CuPy array to use GPU
            wordcloud_array = cp.asarray(wordcloud.to_array())  # Move to GPU
            wordcloud_image = cp.asnumpy(wordcloud_array)  # Convert back to NumPy array for display

            st.write("##### Visualization of Most Frequent Words in All Comments")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud_image, interpolation='bilinear')
            ax.axis('off')  # Hide axes
            st.pyplot(fig)
    st.markdown('---')

    COLumns1, COLumns2 = st.columns([3.5, 1.2])
    with COLumns1:
        positive_comments = len(st.session_state.positive_comments)
        neutral_comments = len(st.session_state.neutral_comments)
        negative_comments = len(st.session_state.negative_comments)
        total_comments = positive_comments + neutral_comments + negative_comments

        if total_comments > 0:
            positive_percentage = (positive_comments / total_comments) * 100
            neutral_percentage = (neutral_comments / total_comments) * 100
            negative_percentage = (negative_comments / total_comments) * 100
        else:
            positive_percentage = neutral_percentage = negative_percentage = 0

        data = pd.DataFrame({
            'Category': ['Comments Analyzed'],  # Single category for stacking
            'Positive': [positive_percentage],
            'Neutral': [neutral_percentage],
            'Negative': [negative_percentage]})

        fig = px.bar(
            data,
            x=['Positive', 'Neutral', 'Negative'],  # Columns to be stacked
            y='Category',  # Single category for horizontal stacking
            orientation='h',
            title='',  # Disable the title
            color_discrete_map={'Positive': '#52ac18', 'Neutral': '#aab7a9', 'Negative': '#c51515'})

        fig.update_traces(
            texttemplate='%{x:.2f}%',  # Format the text as percentage with 2 decimal places
            textposition='inside',  # Position text inside the bars
            insidetextfont=dict(color='black')  # Set text color to black
)
        # Adjust layout to remove gridlines, hide x-axis ticks, and remove x-axis label
        fig.update_layout(
            xaxis=dict(
                showgrid=False,  # Remove gridlines
                showticklabels=False,  # Remove x-axis digits
                title=None  # Remove x-axis label
            ),
            yaxis=dict(
                showgrid=False,
                visible=False  # Hide y-axis
            ),
            barmode='stack',  # Ensure bars are stacked
            bargap=0.7,  # Decrease bar width (reduce gap between bars)
            plot_bgcolor='rgba(0,0,0,0)'  # Remove background color
        )
        st.plotly_chart(fig)
    st.markdown('---')


    COLum1, COLum2 = st.columns([3.5, 1.2])
    with COLum1:
        positive_comments = len(st.session_state.positive_comments)
        neutral_comments = len(st.session_state.neutral_comments)
        negative_comments = len(st.session_state.negative_comments)

        total_comments = positive_comments + neutral_comments + negative_comments

        if total_comments > 0:
            positive_percentage = (positive_comments / total_comments) * 100
            neutral_percentage = (neutral_comments / total_comments) * 100
            negative_percentage = (negative_comments / total_comments) * 100
        else:
            positive_percentage = neutral_percentage = negative_percentage = 0

        data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Percentage': [positive_percentage, neutral_percentage, negative_percentage],
            'Count': [positive_comments, neutral_comments, negative_comments]})

        fig_pie = px.pie(data, names='Sentiment', values='Count',
                         color='Sentiment',
                         color_discrete_map = {'Positive': '#52ac18', 'Neutral': '#aab7a9', 'Negative': '#c51515'})
        st.plotly_chart(fig_pie)
else:
    st.write("### Enter a YouTube video URL to analyze comments!")

# ======================================================================================================================