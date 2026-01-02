"""
추윤서
# 나중에 새로운 레시피를 추가할 때
"""

import chromadb
from sentence_transformers import SentenceTransformer

class RecipeEmbedder:
    def __init__(self, db_path="./modules/vector_db/vectordb_recipes"):
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.client = chromadb.PersistentClient(path=db_path)
        # 우리가 만든 코사인 유사도 컬렉션을 사용
        self.collection = self.client.get_or_create_collection(
            name="recipes_local_cosine",
            metadata={"hnsw:space": "cosine"}
        )

    def add_new_recipe(self, recipe_id, name, ingredients, blog_url):
        """새로운 레시피 하나를 DB에 추가하는 함수"""
        text = f"요리명: {name}, 재료: {ingredients}"
        vector = self.model.encode(text).tolist()
        
        self.collection.add(
            embeddings=[vector],
            metadatas=[{"name": name, "ingredients": ingredients, "blog_url": blog_url}],
            ids=[str(recipe_id)]
        )
        print(f"✅ 새 레시피 '{name}' 추가 완료!")