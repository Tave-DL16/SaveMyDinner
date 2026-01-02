"""
modules.vector_db.search의 Docstring
추윤서
# 벡터 DB 유사도 검색 → 요리명
"""
import chromadb
import json
from sentence_transformers import SentenceTransformer

class RecipeSearcher:
    def __init__(self, db_path="./modules/vector_db/vectordb_recipes"):
        """
        초기화: 로컬 임베딩 모델 로드 및 ChromaDB 연결
        """
        # 1. HuggingFace의 한국어 특화 모델
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 2. ChromaDB 영구 저장소 클라이언트 연결
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 3. reindex.py에서 생성된 recipes_local_cosine 불러오기
        try:
            self.collection = self.client.get_collection(name="recipes_local_cosine")
            print(f"✅ 'recipes_local_cosine' 컬렉션을 성공적으로 로드했습니다. (데이터 수: {self.collection.count()}개)")
        except Exception as e:
            print(f"❌ 컬렉션 로드 실패: {e}")

    def hybrid_search(self, user_ingredients, n_results=5):
        """
        [Hybrid Ranking] 벡터 유사도 + 키워드 매칭 가중치 모델
        """
        # 1. 쿼리 텍스트 생성 및 임베딩 변환
        query_text = " ".join(user_ingredients)
        query_vector = self.model.encode(query_text).tolist()
        
        # 2. 벡터 데이터베이스 검색 (상위 n_results의 2배를 후보군으로 추출)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results * 2
        )
        
        hybrid_results = []
        
        # 3. 검색 결과 루프 및 점수 재계산
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
        
            recipe_ingredients = json.loads(metadata['ingredients'])
            
            # (1) 벡터 유사도 점수 (0~1 범위)
            # ChromaDB의 cosine distance를 유사도로 변환
            vector_score = 1 - results['distances'][0][i]
            
            # (2) 키워드 매칭 점수 (0~1 범위)
            # 사용자가 가진 재료가 실제 레시피에 포함된 비율 계산
            match_count = sum(1 for ing in user_ingredients if ing in recipe_ingredients)
            keyword_score = match_count / len(user_ingredients) if user_ingredients else 0
            
            # (3) 최종 하이브리드 점수 합산 (가중치 설정: 벡터 60%, 키워드 40%)
            final_score = (vector_score * 0.6) + (keyword_score * 0.4)
            
            hybrid_results.append({
                "name": metadata['name'],
                "ingredients": recipe_ingredients,
                "score": round(final_score * 100, 2), # 100점 만점으로 표기
                "url": metadata.get('blog_url', '정보 없음')
            })
            
        # 4. 최종 점수(score) 기준 내림차순 정렬 후 상위 n_results 반환
        return sorted(hybrid_results, key=lambda x: x['score'], reverse=True)[:n_results]

# --- 테스트 코드 ---
if __name__ == "__main__":
    searcher = RecipeSearcher()
    
    # OCR 결과물로 가정할 입력 데이터
    ocr_output = ["콩나물", "마늘", "대파"]
    
    print(f"\n🛒 입력된 식재료: {ocr_output}")
    print("🚀 하이브리드 검색을 시작합니다...\n")
    
    top_recipes = searcher.hybrid_search(ocr_output, n_results=5)
    
    print("="*50)
    print("🍳 AI 추천 레시피 결과")
    print("="*50)
    for idx, r in enumerate(top_recipes, 1):
        print(f"{idx}. {r['name']} (일치율: {r['score']}%)")
        print(f"   주요 재료: {', '.join(r['ingredients'][:5])}...")
        print(f"   레시피 링크: {r['url']}")
        print("-" * 50)
    
    # 다음 모듈(UI)에 넘겨줄 요리명 리스트만 추출
    recipe_names = [r['name'] for r in top_recipes]
    print(f"최종 전달할 요리명 5개: {recipe_names}")