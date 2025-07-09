# scraper/reddit_scraper.py

import os
import praw
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# print("ID:", os.getenv("REDDIT_CLIENT_ID"))
# print("SECRET:", os.getenv("REDDIT_CLIENT_SECRET"))

REDDIT_CLIENT_ID = "CLient_ID"
REDDIT_CLIENT_SECRET = "Client_Secret"
REDDIT_USER_AGENT = "User_agent"

def init_reddit_client():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

def fetch_posts(subreddit_name, limit=500):
    reddit = init_reddit_client()
    subreddit = reddit.subreddit(subreddit_name)

    print(f"Fetching from r/{subreddit_name}...")
    posts = []

    for post in tqdm(subreddit.new(limit=limit)):
        posts.append({
            "id": post.id,
            "title": post.title,
            "selftext": post.selftext,
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
            "subreddit": subreddit_name,
            "url": post.url,
        })

    return posts

def save_posts_to_csv(posts, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(posts)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} posts to {filename}")

if __name__ == "__main__":
    subreddits = ["datascience", "technology"]
    all_posts = []

    for sub in subreddits:
        posts = fetch_posts(sub, limit=300)
        all_posts.extend(posts)

    save_posts_to_csv(all_posts, "data/reddit_posts.csv")
