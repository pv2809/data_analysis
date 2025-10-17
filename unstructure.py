import streamlit as st
import sys
import speech_recognition as sr
from gtts import gTTS
import os
import io
import random
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import spacy
from spacy import displacy

# Ensure required NLTK corpora are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Unstructured Data Analysis", layout="wide")
st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# 🎧 Audio Analysis tab
with tab2:
    st.markdown("### 🎙️ Upload a WAV file to transcribe")
    audio_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

    if audio_file is not None:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            st.info("Transcribing audio...")
            audio_data = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio_data)
                st.success("Transcription complete!")
                st.text_area("Transcribed Text", transcription, height=200)
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Speech Recognition error: {e}")

# 📝 Text Analysis tab
with tab3:
    stories = [
        """During the bustling era of the 1920s, in a city that never slept, Detective Samuel Hart navigated the labyrinthine streets of New York...""",
        """Deep in the Amazon rainforest, a team of scientists embarked on an unprecedented expedition to discover rare medicinal plants..."""
    ]

    if "text_area" not in st.session_state:
        st.session_state.text_area = ""

    if st.button("🎲 Random Story"):
        st.session_state.text_area = random.choice(stories)

    st.session_state.text_area = st.text_area(
        "Paste or modify your text here:",
        value=st.session_state.text_area,
        height=250
    )

    if st.button("Analyze Text 🚀"):
        text = st.session_state.text_area.strip()

        if text:
            blob = TextBlob(text)
            words_and_tags = blob.tags

            nouns = [word for word, tag in words_and_tags if tag.startswith('NN')]
            verbs = [word for word, tag in words_and_tags if tag.startswith('VB')]
            adjectives = [word for word, tag in words_and_tags if tag.startswith('JJ')]
            adverbs = [word for word, tag in words_and_tags if tag.startswith('RB')]

            def make_wordcloud(words, color):
                if not words:
                    st.warning("No words found for this category.")
                    return None
                text_for_wc = " ".join(words)
                wc = WordCloud(width=500, height=400, background_color='black', colormap=color).generate(text_for_wc)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                return fig

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.markdown("### 🧠 Nouns")
                fig = make_wordcloud(nouns, "plasma")
                if fig: st.pyplot(fig)
            with col2:
                st.markdown("### ⚡ Verbs")
                fig = make_wordcloud(verbs, "inferno")
                if fig: st.pyplot(fig)
            with col3:
                st.markdown("### 🎨 Adjectives")
                fig = make_wordcloud(adjectives, "cool")
                if fig: st.pyplot(fig)
            with col4:
                st.markdown("### 💨 Adverbs")
                fig = make_wordcloud(adverbs, "magma")
                if fig: st.pyplot(fig)

            st.markdown("### 📊 POS Counts")
            st.write({
                "Nouns": len(nouns),
                "Verbs": len(verbs),
                "Adjectives": len(adjectives),
                "Adverbs": len(adverbs)
            })

            # Named Entity Recognition
            st.markdown("### 🧠 Named Entity Recognition (NER)")
            doc = nlp(text)
            html = displacy.render(doc, style="ent", jupyter=False)
            st.write("**Detected Entities:**", unsafe_allow_html=True)
            st.markdown(html, unsafe_allow_html=True)

            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                st.markdown("**Entity Table:**")
                st.table(entities)
            else:
                st.info("No named entities found.")
        else:
            st.warning("Please paste or select some text first.")
