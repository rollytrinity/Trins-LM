import streamlit as st
from trigrams import TrigramModel  # Import the trigram model
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from streamlit_d3graph import d3graph, vec2adjmat
from sklearn.manifold import TSNE
import plotly.express as px


st.set_page_config(layout="wide", page_title="Trin's LM Explorer", page_icon=":robot:")

import nltk
nltk.download('punkt_tab')

import gdown

torch.classes.__path__ = []

# Function to get top next word predictions with probabilities
def get_top_predictions(text, num_predictions=100):
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        output = model(input_ids)

    logits = output.logits[0, -1, :]  # Get last token's logits
    probabilities = torch.softmax(logits, dim=0)

    # Get top-k predicted token IDs
    top_k_probs, top_k_indices = torch.topk(probabilities, num_predictions*3)

    # Decode token IDs into words
    words = []
    probs = []

    for idx, prob in zip(top_k_indices, top_k_probs):
        word = tokenizer.decode([idx])  # Decode token into text
        
        # Skip subwords (GPT-2 uses spaces before full words)
        if not word.startswith(" "):  
            continue

        words.append(word.strip())  # Remove leading space
        probs.append(prob.item())

        # Stop once we have enough words
        if len(words) == num_predictions:
            break

    return list(zip(words, probs))

def is_custom_dataset_empty():
    return os.path.exists(custom_dataset_path) and os.path.getsize(custom_dataset_path) == 0

@st.cache_resource
def load_word2vec_model():

    url = "https://drive.google.com/uc?export=download&id=16f4O2RA8M_PaxS09NheAwS4GqZGY4Hqy"
    output = "word2vec-google-news-300.bin"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    from gensim.models import KeyedVectors
    return KeyedVectors.load_word2vec_format(output, binary=True)

@st.cache_resource
def load_model(model_name="local_gpt2"):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model("local_gpt2")

if "embeddings" not in st.session_state:
    st.session_state.embeddings = load_word2vec_model()

st.session_state.city_words = ['Aberdeen', 'Edinburgh', 'Glasgow', 'Inverness', 'Dundee']
st.session_state.animal_words = ['dog', 'cat', 'fish', 'horse', 'cow']

if "your_words" not in st.session_state:
    st.session_state.your_words = []

if "tsne" not in st.session_state:
    vectors = np.array([st.session_state.embeddings[word] for word in st.session_state.city_words + st.session_state.animal_words])
    st.session_state.tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced = st.session_state.tsne.fit_transform(vectors)

# Define dataset path
dataset_options = {"Hunger Games": "data/hunger_games.txt", "Kung Fu Panda": "data/KFP1Script.csv"}
custom_dataset_path = "data/custom_dataset.txt"

# Create a multi-page app with sidebar options as a list

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏠 Home", "🧩 Activities", "📝 Train a Language Model", "➕ Add Sentences", "📚 Words as Vectors", "📊 Visualising Prediction", "🔍 Further Resources"])


with tab1:
    st.title("Welcome to Trin's Fantastic Language Model Explorer")

    st.header("What is this site?")

    st.markdown("""
    Welcome to this interactive hub for learning about **language models**! 🌟
    If you’ve ever wondered how AI like **ChatGPT, Siri, Alexa**, or even **Google** understands your questions, you're in the right place. 
    Language models are designed to understand and generate human-like text, and they’re becoming a huge part of our daily lives — often without us even realizing it!
    
    👾 **Interactive Activities**: Dive in and explore how language models are trained, learn how they interpret context, and even generate their own text based on different inputs. 
    
     🚀 **Why does this matter?** 
    Language models are everywhere now, from the search engines we use to virtual assistants on our phones. Understanding how they work is becoming more important, as they play a huge role in how we interact with technology and access information every day. 
            
                """)
    
    st.header("Where should I start?")
    st.markdown("""
    Start with the **Activities** tab to learn about what language models are and how they work. You will be guided through each of the interactive examples on the site and be able to try them out for yourself, complete with explanations.
    However, if you like, or if you already have some understanding of language models, you can skip straight to experimenting, and come back to the **Activities** page for information as you wish.
    \n
    The **Further Resources** tab has links to videos, articles, and demos that explain language models in more detail, if you want to learn more about them.
    """)
    
    
with tab2:
    st.header("Activities")

    with st.expander("**1) Thinking about Language**"):

        st.markdown("""

        Look at the following sentence and have a think about what word you think would come next.

        -	Summer is hot, winter is _________.\n
        -	She is drinking a cup of _________.
                    
        """)

        if st.button("Show Answers"):
            st.markdown("""
            You might have said:\n
            - Summer is hot, winter is **cold**.\n
            Humans are naturally very clever at understanding language and can easily predict the next word in a sentence.<br><br>
                        
            For the next example, however, maybe you said:
            - She is drinking a cup of **tea**.\n
            - She is drinking a cup of **coffee**.\n
            Or something completely different! This reflects how language can be complex, ambiguous, and confusing, and what kind of things we need to consider when trying to create AI that can understand and generate text.
            """, unsafe_allow_html=True)

    
    with st.expander("**2) What is a Language Model?**"):
        st.markdown("""

        Language models such as ChatGPT work by predicting the next most likely word in a text. Here's ChatGPT's own (correct!) response to what a language model is.
                        """)
            
        st.image("pics/model_explanation.png", width=700)

        st.markdown("If we try asking ChatGPT how it is:")

        st.image("pics/how_are_You.png", width=700)
                    
        st.markdown("This response is the most predictable answer to the question – the language model is predicting the next most likely sequence of words.")

    with st.expander("**3) How does it make predictions?**"):
        st.markdown("""
        In order to predict the next word in a sentence, we need to use a function.
        A function simply takes a number as an input, performs some mathematical operations to it, and then outputs another number.
        For example:

        If $f(x) = x + 2$\n
        $f(0) = 0 + 2 = 2$\n
        $f(1) = 1 + 2 = 3$\n
        $f(2) = 2 + 2 = 4$\n
        $f(3) =$ ____ $=$ __\n

        You can imagine a function as a machine that takes an input (a number) and provides an output (another number):
        
                    """)
        
        st.image("pics/function_machine.png", use_container_width=False)

        st.markdown("""

        A language model is a very complicated function that uses lots of different numbers and performs lots of complex maths, which we aren’t going to worry about! 
                    
        A large language model (LLM) is a type of language model that uses an enormous amount of different numbers in its calculations. ChatGPT is one popular example of this.
                    
        The type of function used by these type of models is called a neural network.

        This function takes a sentence as input, and provides a new sentence as output, which is the words it predicts will come next.

        If you are interested in learning more about what specifically happens in a neural network, you can check out this [video](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown), or try experimenting with [this online demo](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.66222&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false).
        """)

with tab3:
    st.header("Train a Language Model!")
    # Dataset selection
    dataset_options["Custom Dataset"] = custom_dataset_path
    dataset_choice = st.selectbox("Select a dataset:", list(dataset_options.keys()))
    
    use_laplace = False
    alpha = 0.5 

    if dataset_choice == "Custom Dataset":
        use_laplace = True

    # Initialize the model with selected dataset and smoothing option
    trigram_model = TrigramModel(dataset_options[dataset_choice], laplace_smoothing=use_laplace, alpha=alpha)
    
    # Streamlit UI
    
    # Check if the custom dataset is empty
    if is_custom_dataset_empty() and dataset_choice == "Custom Dataset":
        st.warning("The custom dataset is empty! Please add some sentences before generating text.")
        st.stop()  # Stop further execution

    if st.button("Generate Text"):
        generated_text = trigram_model.generate_sentence("")
        st.markdown(f"""
        <div style="font-size: 24px; padding: 20px; border: 1px solid #ddd; background-color: #f4f4f4; color: black; border-radius: 5px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
            {generated_text}
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.header("Add Custom Sentences")
    st.markdown("Enter sentences to add to the custom dataset, which can be used to train the language model.")
    
    user_sentences = st.text_area("Enter your sentences (one per line):")
    
    if st.button("Save Sentences"):
        with open(custom_dataset_path, "a", encoding="utf-8") as file:
            file.write(user_sentences + "\n")
        st.success("Sentences added to the custom dataset!")

    if st.button("Clear All Custom Sentences"):
        open(custom_dataset_path, "w").close()  # This clears the file content
        st.success("All custom sentences have been cleared.")


with tab5:
    st.header("How do we represent words with numbers?")

    embeddings = st.session_state.embeddings

    words = st.session_state.city_words + st.session_state.animal_words + st.session_state.your_words
    vectors = np.array([embeddings[word] for word in words])
    reduced = st.session_state.tsne.fit_transform(vectors)

    df = pd.DataFrame(reduced, columns=["x", "y"])
    df["word"] = words

    groups = (
    ["City"] * len(st.session_state.city_words) +
    ["Animal"] * len(st.session_state.animal_words) +
    ["Custom"] * len(st.session_state.your_words)
    )   
    df["group"] = groups

    # Plot with Plotly
    fig = px.scatter(df, x="x", y="y", text="word", color="group",)
    fig.update_traces(marker=dict(size=12))
    fig.update_traces(textposition='top center')
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_xaxes(showgrid=True, zeroline=False, visible=True)
    fig.update_yaxes(showgrid=True, zeroline=False, visible=True)

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    new_word = st.text_input("Enter a new word to see its vector representation:")

    if new_word != "" and new_word not in st.session_state.your_words:
        st.session_state.your_words.append(new_word)
        st.rerun()

    st.markdown("__________________________________________________________________")
    st.markdown("""
    The graph above shows numerical representations of words. The coordinates of each of the words provide information about those words, which can then be input to our language models.
    Notice how the 'city' words, and the 'animal' words are grouped together. The coordinates therefore tell us something about the meaning of the words - in this case how 'city-ish' or 'animal-ish' they are.
    These coordinates can be treated as groups of numbers, called vectors. 
    \n
    You can add your own words to the graph too. It's important to note that the coordinates we are using here are squeezed down from a higher dimension.
    This sounds confusing, but all this means is that in reality there are many more coordinates than two, allowing more word information to be described. But it's easier to understand when we can see it in two dimensions, so every time a new word is entered,
    we recalculate the coordinates of all the words in two dimensions.
    \n
                """)
    
    st.subheader("Further Resources")
    st.markdown(
    """
    -   [A more detailed and complex version of this visualisation](https://projector.tensorflow.org/)
    """
    )
        

with tab6:
    # Ensure model is in evaluation mode
    model.eval()

    # Initialize sentence with a starting token
    if 'sentence' not in st.session_state:
        st.session_state.sentence = ["Hello"]

    # Streamlit Interface
    st.header("Visualising Prediction")

    # Display the sentence
    st.session_state.sentence = st.text_input(label='Enter a sentence:')
    st.session_state.sentence = st.session_state.sentence.split()

    # Get the next word probabilities using the actual model
    if len(st.session_state.sentence) > 0:
        predictions = get_top_predictions(" ".join(st.session_state.sentence), num_predictions=10)

        # Get next word probabilities
        words, probabilities = zip(*predictions)
        print(predictions)
        prev_word = [st.session_state.sentence[-1]]
        words = list(words)
        probabilities = [round(x,2) for x in probabilities]

        # Source node names
        source = prev_word * len(words)
        # Target node names
        target = words[:]
        # Edge Weights
        weight = probabilities

        words = [x for x in words if x not in prev_word]

        for word in words:
            predictions = get_top_predictions(" ".join(st.session_state.sentence) + " " + word, num_predictions=10)

            if len(predictions) > 0:
                next_words, next_probabilities = zip(*predictions)

                next_probabilities = [round(x,2) for x in next_probabilities]
                next_words = list(next_words)

                source += [word] * len(next_words)
                target += next_words
                weight += next_probabilities


        # Convert the vector into a adjacency matrix
        adjmat = vec2adjmat(source, target, weight=weight)

        # Initialize
        d3 = d3graph(slider=None, charge=1000)

        # Process adjmat
        d3.graph(adjmat, color=None, cmap='viridis')

        unique_nodes = list(d3.node_properties.keys())

        labels = []
        sizes = []

        for node in unique_nodes:
            if node == st.session_state.sentence[-1]:
                labels.append('#B8A0D0')  # Root node
                sizes.append(3)
            elif node in words:
                labels.append('#88C9B2')  # First layer
                sizes.append(2)
            else:
                labels.append('#F2D48A')  # Second layer
                sizes.append(1)

        d3.set_node_properties(color=labels, size=sizes)

        # Plot
        d3.show(show_slider=False)

    st.subheader("About")
    st.markdown("""
    The graph above visualises the next most likely words based on your inputs.\n""")

    st.markdown("<p style='color: #B8A0D0;'>The purple word is the last word you entered.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #88C9B2;'>The green words are likely to follow the purple word, continuing your sentence.</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #F2D48A;'>The yellow words are likely to follow green words.</p>", unsafe_allow_html=True)

    st.markdown("""
    The thickness of lines between words indicates the likelihood of that sentence.
    This example is powered using GPT-2, an example of a large language model. It is a neural network that has been trained on a large amount of text data to predict the next word in a sentence.
    The model must use this information to decide which words should come next.
    """)

with tab7:
    st.header("Further Resources")

    st.markdown("""ThreeBlueBrown is a YouTube channel that explains complex maths and AI concepts in an easy-to-understand way. Some videos are more complex than others, but all are super useful for developing intuition around these topics.\n""")
    st.markdown("""
                - [What is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown)
                - [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs&ab_channel=3Blue1Brown)
                """)
    
    st.markdown("""\nIf you've enjoyed experimenting with the live demos on this site, the following will also be interesting to you! They all have variable complexity, so feel free to pick and choose according to your knowledge and experience.""")
    st.markdown("""
                - [Machine Learning for Kids - a set of tutorials for machine learning, aimed at kids but interesting for anyone!](https://machinelearningforkids.co.uk/)
                - [Teachable Machine - a set of tutorials for machine learning. Less centred around language but teaches fundamentals of AI which are completely relevant!](https://teachablemachine.withgoogle.com/)
                - [Word embedding visualisation](https://projector.tensorflow.org/)
                - [Tensorflow playground, for exploring the internals of neural nets](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.88734&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
                """)