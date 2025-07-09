

import pandas as pd
import re
import emoji
import spacy
from unidecode import unidecode
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Combine emojis into text (optional)
    text = emoji.demojize(text)

    # Remove markdown links: [text](url)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalize accents and emojis
    text = unidecode(text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def preprocess_dataframe(df):
    tqdm.pandas()

    # Merge title + body
    df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")

    # Clean text
    df["clean_text"] = df["text"].progress_apply(clean_text)

    # Lemmatize (optional for BERTopic; useful for LDA)
    df["lemmatized_text"] = df["clean_text"].progress_apply(lemmatize_text)

    return df

if __name__ == "__main__":
    df = pd.read_csv("/Users/eeshanpancholiya/Desktop/personal_project/trending_topics/data/reddit_posts.csv")
    df_clean = preprocess_dataframe(df)
    df_clean.to_csv("/Users/eeshanpancholiya/Desktop/personal_project/trending_topics/data/reddit_posts_clean.csv", index=False)
    print(" Saved cleaned file: /Users/eeshanpancholiya/Desktop/personal_project/trending_topics/data/reddit_posts_clean.csv")
