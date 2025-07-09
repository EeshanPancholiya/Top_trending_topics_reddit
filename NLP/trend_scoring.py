

import pandas as pd
from collections import Counter
from datetime import datetime, timedelta, timezone

def compute_velocity(df, time_window_hours=24):
    now = datetime.now(timezone.utc)  # timezone-aware
    recent_cutoff = now - timedelta(hours=time_window_hours)

    recent_posts = df[df['timestamp'] >= recent_cutoff]  # now both sides are tz-aware
    velocity = recent_posts.groupby("topic").size().rename("velocity")
    return velocity

def compute_spread(df):
    spread = df.groupby("topic")["subreddit"].nunique().rename("spread")
    return spread

def compute_trend_score(df, velocity_weight=0.6, spread_weight=0.4):
    velocity = compute_velocity(df)
    spread = compute_spread(df)

    trend_df = pd.concat([velocity, spread], axis=1).fillna(0)

    # Normalize scores to 0–1
    trend_df["velocity_norm"] = trend_df["velocity"] / trend_df["velocity"].max()
    trend_df["spread_norm"] = trend_df["spread"] / trend_df["spread"].max()

    # Weighted trend score
    trend_df["trend_score"] = (
        velocity_weight * trend_df["velocity_norm"] +
        spread_weight * trend_df["spread_norm"]
    )

    trend_df = trend_df.sort_values("trend_score", ascending=False)
    trend_df.reset_index(inplace=True)
    return trend_df

if __name__ == "__main__":
    df = pd.read_csv("data/reddit_topics_labeled.csv")
    df["timestamp"] = pd.to_datetime(df["created_utc"])

    trend_scores = compute_trend_score(df)
    trend_scores.to_csv("data/trending_topics_scores.csv", index=False)

    print("✅ Saved: trending_topics_scores.csv")
    print(trend_scores.head())


