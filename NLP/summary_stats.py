import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import datetime

# --- CONFIG ---
OPENAI_API_KEY = "Use your API Key"  # or set directly here
client = OpenAI(api_key=OPENAI_API_KEY)

def load_data():
    scores = pd.read_csv("data/trending_topics_scores.csv")
    topics = pd.read_csv("data/topic_keywords.csv")
    posts = pd.read_csv("data/reddit_topics_labeled.csv")
    posts['timestamp'] = pd.to_datetime(posts['created_utc'])
    return scores, topics, posts

def generate_explanation(keywords, sample_titles):
    prompt = f"""
You are a social media trend analyst. Given the keywords and Reddit post titles below, explain in 1-2 sentences why this topic might be trending:

Keywords: {', '.join(keywords)}
Post Titles: {sample_titles}

Your explanation:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Explanation unavailable due to API error."

def summarize_topic(topic_row, posts_df):
    posts_df.columns = posts_df.columns.str.lower()
    topic_id = topic_row['topic']
    topic_posts = posts_df[posts_df['topic'] == topic_id].copy()
    top_posts = topic_posts.sort_values("timestamp", ascending=False).head(5)
    
    # LLM Explanation
    sample_titles = "; ".join(top_posts["title"].tolist())[:500]
    keywords = eval(topic_row['representation']) if isinstance(topic_row['representation'], str) else topic_row['representation']
    explanation = generate_explanation(keywords, sample_titles)

    # Time series count
    topic_posts['day'] = topic_posts['timestamp'].dt.date
    daily_counts = topic_posts.groupby("day").size()

    return {
        "Topic ID": topic_id,
        "Name": topic_row.get("name", f"Topic {topic_id}"),
        "Keywords": keywords,
        "Velocity": topic_row["velocity"],
        "Spread": topic_row["spread"],
        "Trend Score": round(topic_row["trend_score"], 2),
        "Top Posts": top_posts[["title", "url"]],
        "Explanation": explanation,
        "Time Series": daily_counts
    }

def main():
    st.set_page_config(page_title="Trending Topics Summary", layout="wide")
    
    st.title("Most Popular Data Science & Technology Reddit Topics Today")

    scores, topics, posts = load_data()
    topics.columns = topics.columns.str.lower()
    merged = scores.merge(topics, on="topic", how="left")

    merged_sorted = merged.sort_values("trend_score", ascending=False).reset_index(drop=True)

    for i, row in merged_sorted.iterrows():
        summary = summarize_topic(row, posts)

        # Prepare a string for keywords preview, e.g. top 3 keywords
        keywords_preview = ", ".join(summary["Keywords"][:3]) if summary["Keywords"] else "No keywords"

        # Compose expander title with ranking + keywords or topic name
        if i == 0:
            topic_heading = f" #1 Most Popular Topic Today: {summary['Name']} ({keywords_preview})"
        elif i == 1:
            topic_heading = f" #2 Most Popular Topic Today: {summary['Name']} ({keywords_preview})"
        elif i == 2:
            topic_heading = f"#3 Most Popular Topic Today: {summary['Name']} ({keywords_preview})"
        else:
            topic_heading = f"Topic #{i+1}: {summary['Name']} ({keywords_preview})"

        with st.expander(f"{topic_heading} — Trend Score: {summary['Trend Score']}"):
            # rest remains the same ...
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Why it might be trending:**")
                st.markdown(f"> {summary['Explanation']}")

                st.markdown("** Top Reddit Threads:**")
                for _, post in summary["Top Posts"].iterrows():
                    st.markdown(f"- [{post['title']}]({post['url']})")

            with col2:
                st.markdown("**Topic Metrics**")
                st.markdown(f"**Velocity:** {summary['Velocity']}  \n"
                            f"**Spread:** {summary['Spread']}  \n"
                            f"**Trend Score:** {summary['Trend Score']}")
                st.caption(" Velocity = rate of increase in mentions. Spread = how many users participated.")

                fig, ax = plt.subplots()
                ts = summary["Time Series"]
                ts.rolling(2).mean().plot(ax=ax, color="tab:blue")
                ax.set_title("Mentions Over Time")
                ax.set_ylabel("Mentions")
                ax.set_xlabel("Date")
                ax.grid(True)
                st.pyplot(fig)

    st.markdown("---")  # horizontal line separator
    st.header(" Glossary: Understanding the Metrics")

    st.markdown("""
                - **Trend Score:** A combined metric that reflects the overall popularity and growth of the topic, taking into account velocity, spread, and possibly other factors.
                - **Velocity:** The rate at which mentions of the topic are increasing over time — how fast the topic is gaining traction.
                - **Spread:** The breadth of participation, i.e., how many unique users or sources are discussing the topic.
                - **Keywords:** The main terms or phrases that represent the topic, extracted from related posts.
                - **Top Reddit Threads:** The most recent and relevant Reddit discussions that exemplify the topic.
                - **Mentions Over Time:** A time series plot showing the number of times the topic has been mentioned daily, smoothed for clarity.
                """)          
                





if __name__ == "__main__":
    main()
