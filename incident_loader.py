import pandas as pd

def load_incidents(csv_path):
    df = pd.read_csv(csv_path)
    records = []

    for _, r in df.iterrows():
        # incident_loader.py
        records.append({
            "source": "INCIDENT",
            "id": r.get("Number"),
            "title": r.get("Short description", ""),
            "content": r.get("Root Cause", ""),
            "resolution": r.get("Resolution", ""),
            "rfc": r.get("RFC", ""),
            "kb_ref": r.get("Attached Knowledge", "")
        })

    return records
