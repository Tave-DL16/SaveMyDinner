import chromadb
import pandas as pd

PERSIST_DIR = "./modules/vector_db/vectordb_recipes"
COLLECTION_NAME = "recipes_1000"
OUTPUT_CSV = "./data/recipes_chroma_dump.csv"

client = chromadb.PersistentClient(path=PERSIST_DIR)
col = client.get_collection(COLLECTION_NAME)

print(f"Collection count: {col.count()}")

# 전부 가져오기 (데이터 많으면 limit/offset 방식으로 나눠도 됨)
data = col.get(
    include=["documents", "metadatas"]
)

rows = []
for _id, doc, meta in zip(
    data["ids"],
    data["documents"],
    data["metadatas"]
):
    row = {
        "id": _id,
        "document": doc,
    }

    # metadata 펼치기
    if isinstance(meta, dict):
        for k, v in meta.items():
            row[f"meta_{k}"] = v

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"CSV exported: {OUTPUT_CSV}")