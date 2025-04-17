import streamlit as st
from trigrams import TrigramModel  # Import the trigram model
import os
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from streamlit_d3graph import d3graph, vec2adjmat
from sklearn.manifold import TSNE
import plotly.express as px
import logging
import nltk
import gdown


st.title("✅ Hello Streamlit!")
st.write("If you see this, the app is running properly!")

if st.button("Click me"):
    st.success("Button works!")
