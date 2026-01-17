import chromadb
from pprint import pprint

PERSIST_DIR = "./modules/vector_db/vectordb_recipes"

client = chromadb.PersistentClient(path=PERSIST_DIR)

cols = client.list_collections()
print("Collections:")
for c in cols:
    print("-", c.name)

if cols:
    col = client.get_collection(cols[0].name)
    print("\nCount:", col.count())

    sample = col.get(
        include=["documents", "metadatas"],
        # include=["documents", "metadatas", "embeddings"],
        limit=5
    )
    pprint(sample)
else:
    print("No collections found.")