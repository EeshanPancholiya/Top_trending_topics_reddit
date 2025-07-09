

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

def load_data(filepath="data/reddit_posts_clean.csv"):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["created_utc"])
    return df

def run_topic_modeling(texts, timestamps):
    topic_model = BERTopic(
        language="english",
        min_topic_size=15,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, probs

def save_topic_output(df, topics, probs, topic_model):
    df["topic"] = topics
    df["probability"] = probs

    # Save summary
    df.to_csv("data/reddit_topics_labeled.csv", index=False)

    # Save topic keywords
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("data/topic_keywords.csv", index=False)

    print("✅ Saved: reddit_topics_labeled.csv and topic_keywords.csv")

if __name__ == "__main__":
    df = load_data()
    texts = df["lemmatized_text"].tolist()
    timestamps = df["timestamp"].tolist()

    topic_model, topics, probs = run_topic_modeling(texts, timestamps)
    save_topic_output(df, topics, probs, topic_model)

    # Optional: visualize in browser
    topic_model.visualize_topics().show()
