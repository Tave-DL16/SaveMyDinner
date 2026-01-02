"""
추윤서
# 차원 불일치 해결
# 기존 DB (1536차원) -> 로컬 모델(768차원) 으로 다시 임베딩

# 거리 측정 방식 : 코사인 유사도
"""
import chromadb
import json
from sentence_transformers import SentenceTransformer

# 1. 설정
DB_PATH = "./modules/vector_db/vectordb_recipes"
NEW_COL_NAME = "recipes_local_cosine"

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
client = chromadb.PersistentClient(path=DB_PATH)

# 2. 기존 데이터 (recipes_1000)
old_col = client.get_collection(name="recipes_1000")
all_data = old_col.get()

# 3. 초기화
try:
    client.delete_collection(NEW_COL_NAME)
except:
    pass

# 4. 새 컬렉션 생성 (거리 측정 방식을 'cosine'으로)
new_col = client.create_collection(
    name=NEW_COL_NAME,
    metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용 설정
)

# 5. 재임베딩 및 저장
print(f"🚀 {NEW_COL_NAME} 컬렉션 생성 및 재색인 시작...")
for i in range(len(all_data['ids'])):
    metadata = all_data['metadatas'][i]
    text_to_embed = f"요리명: {metadata['name']}, 재료: {metadata['ingredients']}"
    
    vector = model.encode(text_to_embed).tolist()
    
    new_col.add(
        embeddings=[vector],
        metadatas=[metadata],
        ids=[all_data['ids'][i]]
    )
    if (i+1) % 200 == 0:
        print(f"✅ {i+1}개 완료...")

print(f"✨ 완료! 이제 search.py에서 '{NEW_COL_NAME}'을 사용하세요.")