import faiss, pickle, os
from offline_embedder import OfflineEmbedder
from kb_loader import load_kb
from incident_loader import load_incidents

embedder = OfflineEmbedder()

# Ensure the directory exists
os.makedirs("faiss_index", exist_ok=True)

kb = load_kb("data/kb_knowledge.csv")
inc = load_incidents("data/incidents_raw.csv")

documents = kb + inc
texts = []

for d in documents:
    if d["source"] == "KB":
        texts.append(
            f"KB_ID: {d.get('id','')}\nTitle: {d.get('title','')}\nContent: {d.get('content','')}"
        )
    else:
        texts.append(
            f"[INCIDENT]\nINC_ID: {d.get('id','')}\nTitle: {d.get('title','')}\n"
            f"Linked_CHG: {d.get('change','')}\nLinked_KB: {d.get('kb_ref','')}\n"
            f"Issue: {d.get('content','')}\nResolution: {d.get('resolution','')}"
        )


# Generate embeddings
embeddings = embedder.embed(texts)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Write FAISS index AFTER it exists
faiss.write_index(index, "faiss_index/unified.index")

# Save metadata
with open("faiss_index/metadata.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Unified Incident +  KB FAISS index created")
