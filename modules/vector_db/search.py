"""
modules.vector_db.search
작성자: 추윤서
기능: 자취생/1인 가구 맞춤형 레시피 정제 및 중복 제거 검색 엔진
"""
import chromadb
import json
import re
# import os  # OpenAI 모델 사용 시 필요
from sentence_transformers import SentenceTransformer
# from openai import OpenAI  # OpenAI 모델 사용 시
# from dotenv import load_dotenv  # OpenAI API 키 사용 시 필요

# load_dotenv()  # OpenAI API 키 사용 시 필요

class RecipeSearcher:
    def __init__(self, db_path="./modules/vector_db/vectordb_recipes"):
        """
        초기화: 로컬 임베딩 모델 로드 및 ChromaDB 연결
        """

        # 1. HuggingFace의 한국어 특화 모델 (768차원)
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')

        # # 1. OpenAI 클라이언트 (1536차원)
        # api_key = os.getenv('OPENAI_API_KEY')
        # if api_key:
        #     self.openai_client = OpenAI(api_key=api_key)
        # else:
        #     print("⚠️ OPENAI_API_KEY not found - VectorDB search will be limited")
        #     self.openai_client = None

        # 2. ChromaDB 클라이언트 연결
        self.client = chromadb.PersistentClient(path=db_path)

        # 3. 컬렉션 로드
        try:
            # recipes_local_cosine 컬렉션 우선 로드 (로컬 모델용)
            self.collection = self.client.get_collection(name="recipes_local_cosine")
            print(f"✅ 'recipes_local_cosine' 컬렉션 로드 완료. (데이터: {self.collection.count()}개)")

            # # 먼저 recipes_1000 시도 (temp 데이터)
            # try:
            #     self.collection = self.client.get_collection(name="recipes_1000")
            #     print(f"✅ 'recipes_1000' 컬렉션 로드 완료. (데이터: {self.collection.count()}개)")
            # except:
            #     # 없으면 recipes_local_cosine 시도 (기존 이름)
            #     self.collection = self.client.get_collection(name="recipes_local_cosine")
            #     print(f"✅ 'recipes_local_cosine' 컬렉션 로드 완료. (데이터: {self.collection.count()}개)")
        except Exception as e:
            print(f"❌ 컬렉션 로드 실패: {e}")

    def clean_recipe_name(self, name):
        """
        [고도화된 정제] 자취생용 수식어 제거 및 단어 정렬 정규화
        """
        name = re.sub(r'[^\w\s]', ' ', name).lower()
        
        # 1인 가구에게 노이즈가 되는 단어들 대폭 제거
        stop_words = [
            '레시피', '만들기', '방법', '황금레시피', '간단', '초간단', '아삭한', '맛있는', 
            '꿀팁', '집밥', '반찬', '양념', '젓국', '하얀', '식감이', '매력적인', '단짠', 
            '입맛돋궈주는', '새콤아삭', '든든한', '최고의'
        ]
        
        words = name.split()
        # 어순 정규화: 단어를 가나다순으로 정렬하여 '하얀 콩나물'과 '콩나물 하얀'을 동일하게 처리
        cleaned_words = sorted([w for w in words if w not in stop_words])
        
        unique_words = []
        for w in cleaned_words:
            if w not in unique_words:
                unique_words.append(w)
                
        return " ".join(unique_words).strip()

    def is_too_similar(self, new_name, existing_names, threshold=0.6):
        """
        [중복도 검사] Overlap Coefficient를 사용하여 비슷한 메뉴 중복 방지
        """
        new_set = set(new_name.split())
        if not new_set: return True

        for existing in existing_names:
            existing_set = set(existing.split())
            if not existing_set: continue
            
            # 단어 중복 비율 계산
            intersection = new_set.intersection(existing_set)
            overlap = len(intersection) / min(len(new_set), len(existing_set))
            
            if overlap >= threshold:
                return True
        return False

    def get_embedding(self, text: str) -> list:
        """로컬 SentenceTransformer 모델로 임베딩 생성 (768차원)"""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
            return None

    # def get_embedding(self, text: str) -> list:
    #     """OpenAI API로 임베딩 생성 (1536차원)"""
    #     if self.openai_client is None:
    #         return None
    #
    #     try:
    #         response = self.openai_client.embeddings.create(
    #             model="text-embedding-3-small",
    #             input=text
    #         )
    #         return response.data[0].embedding
    #     except Exception as e:
    #         print(f"임베딩 생성 실패: {e}")
    #         return None

    def hybrid_search(self, user_ingredients, n_results=5):
        """
        벡터 유사도(60%) + 키워드 매칭(40%) + 자취생용 다양성 필터
        """
        # 로컬 모델용: reindex.py와 동일한 형식으로 쿼리 생성
        ingredients_text = ", ".join(user_ingredients)
        query_text = f"요리명: , 재료: {ingredients_text}"
        query_vector = self.get_embedding(query_text)

        # # OpenAI 모델용: DB 구축 시 사용한 형식과 동일하게 쿼리 생성
        # ingredients_text = ", ".join(user_ingredients)
        # query_text = f"재료: {ingredients_text}. 이 재료들로 만들 수 있는 요리를 찾아주세요."
        # query_vector = self.get_embedding(query_text)

        if query_vector is None:
            print("⚠️ 임베딩 생성 실패, 빈 결과 반환")
            return []
        
        # 중복을 걸러내고도 5개를 채우기 위해 충분한 후보(75개) 추출
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results * 15 
        )
        
        hybrid_results = []
        final_names = [] 

        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            raw_name = metadata['name']
            cleaned_name = self.clean_recipe_name(raw_name)
            
            # 기존 결과와 너무 비슷하면 건너뜀 (다양성 확보)
            if self.is_too_similar(cleaned_name, final_names):
                continue
            
            recipe_ingredients = json.loads(metadata['ingredients'])
            vector_score = 1 - results['distances'][0][i]
            match_count = sum(1 for ing in user_ingredients if ing in recipe_ingredients)
            keyword_score = match_count / len(user_ingredients) if user_ingredients else 0
            
            final_score = (vector_score * 0.6) + (keyword_score * 0.4)
            
            hybrid_results.append({
                "name": cleaned_name,
                "original_name": raw_name,
                "score": round(final_score * 100, 2),
                "ingredients": recipe_ingredients,
                "url": metadata.get('blog_url', '정보 없음')
            })
            final_names.append(cleaned_name)
            
            if len(hybrid_results) == n_results:
                break
        
        return hybrid_results

# --- 테스트 및 통합용 출력 코드 ---
if __name__ == "__main__":
    searcher = RecipeSearcher()
    ocr_output = ["콩나물", "마늘", "대파"]
    
    print(f"\n🛒 [자취생 모드] 인식된 식재료: {ocr_output}")
    print("🚀 중복 없는 고도화된 검색을 시작합니다...\n")
    
    top_recipes = searcher.hybrid_search(ocr_output, n_results=5)
    
    print("="*50)
    print("🍳 SaveMyDinner: 오늘의 추천 레시피")
    print("="*50)
    for idx, r in enumerate(top_recipes, 1):
        print(f"{idx}. {r['name']} (적합도: {r['score']}%)")
        print(f"   [출처: {r['original_name']}]")
        print(f"   🔗 {r['url']}")
        print("-" * 50)
    
    # 수민이에게 전달할 최종 요리명 리스트 추출 및 출력
    recipe_names = [r['name'] for r in top_recipes]
    print(f"✅ 최종 전달할 정제된 요리명 5개: {recipe_names}")