import pandas as pd
from bs4 import BeautifulSoup

def clean_html(text):
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "lxml").get_text(" ", strip=True)

def load_kb(csv_path):
    # print(csv_path)
    # df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, encoding='latin-1')
    records = []

    for _, r in df.iterrows():
        records.append({
            "source": "KB",
            "id": r.get("KB_Number", ""),
            "title": r.get("KB_Title", ""),
            "content": clean_html(r.get("KB_Description", ""))
        })

    return records
