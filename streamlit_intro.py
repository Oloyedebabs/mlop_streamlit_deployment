import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os
import torch
from transformers import pipeline

import boto3
# S3 setup
bucket_name = "mlops-oba"
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

# Initialize S3 client
s3 = boto3.client('s3')

# Function to download model from S3
def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                s3.download_file(bucket_name, s3_key, local_file)

# Page configuration
st.set_page_config(page_title="Sentiment Analysis Demo", page_icon="🧠", layout="wide")

# Header section
st.title("🧠 Sentiment Analysis with TinyBERT")
st.markdown("Welcome to the Sentiment Analysis Model Demo! This tool helps analyze tweet sentiment and determine potential natural disaster indications in specific areas.")

# Sidebar for model options
st.sidebar.header("Model Options")
model_type = st.sidebar.selectbox("Select Model", ["TinyBERT Sentiment", "DistilBERT Sentiment"])

# Button to download model from S3
if st.sidebar.button("Get Model"):
    with st.spinner("Downloading model..."):
        download_dir(local_path, s3_prefix)
    st.sidebar.success("Model downloaded successfully!")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.write("Choose the model to use for sentiment analysis.")

# Text input
st.header("🔍 Enter Text for Analysis")
user_input = st.text_area("Enter text here:", placeholder="Type some text...")

# Device setup for model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load classifier
try:
    classifier = pipeline('text-classification', model=local_path, device=device)
except Exception as e:
    st.error("Model not found locally. Please click 'Get Model' to download.")

# Analyze sentiment on button click
if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing..."):
            output = classifier(user_input)
            result = output[0]  # Get the first result from the output list

            # Display results
            st.success("Analysis Complete!")
            col1, col2 = st.columns(2)
            col1.metric("Sentiment Score", f"{result['score']:.2f}")
            col2.write(f"**Sentiment Label**: {result['label']}")

            # Add icons based on sentiment
            if result["label"] == "general":
                col2.write("✅ No Disaster!")
            elif result["label"] == "disaster":
                col2.write("🚨 Disaster Alert!")
            else:
                col2.write("😐 Neutral")
    else:
        st.warning("Please enter some text for analysis.")

# Footer
st.markdown("""
---
Made with ❤️ by OBA | Powered by [Streamlit](https://streamlit.io) & [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
""")
