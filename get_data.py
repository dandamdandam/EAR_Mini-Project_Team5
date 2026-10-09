from dotenv import load_dotenv
import os

load_dotenv(".env", override=True)

KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
import kagglehub

# Download latest version
path = kagglehub.competition_download('2026-ieee-big-data-traffic-flow-bench')

print("Path to competition files:", path)
with open(".env", "a") as f:
    f.write(f"\nFILE_PATH={path}")