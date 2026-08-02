import pandas as pd
from pathlib import Path

DATASET = Path("dataset")

files = [
    "messages.csv",
    "message_history.csv",
    "message_events.csv",
    "users.csv",
    "groups.csv",
    "group_members.csv",
    "business_accounts.csv",
    "user_business_history.csv",
    "images.csv",
    "voice_notes.csv",
    "daily_notification_summary.csv",
    "sample_messages.csv"
]

for file in files:
    print("\n" + "=" * 80)
    print(file)
    print("=" * 80)

    df = pd.read_csv(DATASET / file)

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nFirst 3 rows:")
    print(df.head(3))

    print("\nMissing values:")
    print(df.isnull().sum())
    