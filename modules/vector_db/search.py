"""
modules.vector_db.search
작성자: 추윤서
기능: 자취생/1인 가구 맞춤형 레시피 정제 및 중복 제거 검색 엔진
"""
import chromadb
import json
import re
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class RecipeSearcher:
    def __init__(self, db_path="./modules/vector_db/vectordb_recipes"):
        """
        초기화: 로컬 임베딩 모델 로드 및 ChromaDB 연결
        """

        # 1. HuggingFace의 한국어 특화 모델 (768차원)
        self.model = SentenceTransformer('jhgan/ko-sroberta-multitask', device='cpu')

        # 2. OpenAI API 설정
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            print("✅ OpenAI API 연결 완료")
        else:
            print("⚠️ OPENAI_API_KEY not found - LLM 정제 기능이 제한됩니다")
            self.openai_client = None

        # 3. ChromaDB 클라이언트 연결
        self.client = chromadb.PersistentClient(path=db_path)

        # 4. 컬렉션 로드
        try:
            self.collection = self.client.get_collection(name="recipes_local_cosine")
            print(f"✅ 'recipes_local_cosine' 컬렉션 로드 완료. (데이터: {self.collection.count()}개)")
        except Exception as e:
            print(f"❌ 컬렉션 로드 실패: {e}")
    
    def clean_with_llm(self, raw_name):
        """OpenAI API를 사용하여 환각 현상을 방지하고 핵심 요리명만 정확히 추출"""
        
        # OpenAI를 사용할 수 없으면 규칙 기반으로 대체
        if self.openai_client is None:
            print(f"⚠️ OpenAI 미사용: '{raw_name}' -> 규칙 기반 처리")
            return self.clean_recipe_name(raw_name)
        
        # OpenAI API용 프롬프트 (더 명확한 지시사항)
        prompt = f"""레시피 제목에서 핵심 요리명만 추출하세요.

제거할 것:
- 숫자, 날짜, 에피소드 번호
- 수식어(맛있는, 간단한, 아삭한, 입맛돋구는 등)
- 조리방법 관련 단어(만드는법, 레시피, 만들기, 황금레시피 등)
- 특수문자(!,.,.. 등)

예시:
입력: [176.오트밀과일빵(2025.11.7)]
출력: 오트밀과일빵

입력: [[만개백과] EP. 18 가끔 생각나는 야채샐러드빵]
출력: 야채샐러드빵

입력: [에어프라이어 요리 양파햄치즈빵 만드는 법 너무 맛있잖아]
출력: 양파햄치즈빵

입력: [아삭한 콩나물무침 레시피 만들기]
출력: 콩나물무침

입력: [입맛 돋구는 양파덮밥 레시피!]
출력: 양파덮밥

입력: [{raw_name}]
출력:"""

        try:
            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 요리 명칭 정제 전문가입니다. 핵심 요리명만 추출하고, 수식어와 조리방법 관련 단어는 모두 제거합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.0,
            )
            
            refined = response.choices[0].message.content.strip()
            
            # 후처리: 불필요한 텍스트 및 특수문자 제거
            refined = re.sub(r'출력:|결과:|->|:|\*|```|!|\.|…', '', refined).strip()
            refined = refined.split('\n')[0].strip()
            
            # 추가 정제: 남은 불필요한 단어 제거
            noise_words = ['레시피', '만들기', '만드는법', '황금레시피']
            for word in noise_words:
                refined = refined.replace(word, '').strip()
            
            # 검증: 결과가 유효한지 확인
            if refined and len(refined) >= 2 and not refined.isdigit():
                print(f"✅ OpenAI 정제: '{raw_name}' -> '{refined}'")
                return refined
            else:
                print(f"⚠️ OpenAI 결과 불량: '{refined}' -> 규칙 기반으로 대체")
                return self.clean_recipe_name(raw_name)
            
        except Exception as e:
            print(f"⚠️ OpenAI API 오류: {e}")
            print(f"   '{raw_name}' -> 규칙 기반 처리")
            return self.clean_recipe_name(raw_name)

    def clean_recipe_name(self, name):
        """
        [고도화된 정제] 자취생용 수식어 제거 (어순 유지)
        """
        # 1. 특수문자를 공백으로 변환 (괄호, 대괄호, 점 등)
        name = re.sub(r'[\[\]().,\-_]', ' ', name)
        
        # 2. 숫자와 날짜 패턴 제거 (예: 176, 2025.11.7, EP 18)
        name = re.sub(r'\d+\.?\d*\.?\d*', ' ', name)
        name = re.sub(r'ep\s*\d+|episode\s*\d+', ' ', name, flags=re.IGNORECASE)
        
        # 3. 소문자 변환
        name = name.lower()
        
        # 4. 노이즈 단어 제거
        stop_words = [
            '레시피', '만들기', '방법', '황금레시피', '간단', '초간단', '아삭한', '맛있는', 
            '꿀팁', '집밥', '반찬', '양념', '젓국', '하얀', '식감이', '매력적인', '단짠', 
            '입맛돋궈주는', '새콤아삭', '든든한', '최고의', '끓이는법', '끓이기', '조리법',
            '요리법', '쉬운', '빠른', '특급', '비법', '황금', '꿀', '백선생', '알토란',
            '입맛', '돋구는', '간단하지만', '특별한', '영양만점', '초스피드', '속성',
            '만개백과', '가끔', '생각나는', '너무', '맛있잖아', '에어프라이어', '요리',
            '만드는', '법', '법', 'ep', 'episode'
        ]
        
        words = name.split()
        # 어순 유지하면서 불용어만 제거
        cleaned_words = [w for w in words if w.strip() and w not in stop_words]
        
        # 중복 제거하되 순서는 유지
        unique_words = []
        for w in cleaned_words:
            if w not in unique_words and len(w) > 0:
                unique_words.append(w)
        
        result = " ".join(unique_words).strip()
        
        # 결과가 비어있으면 원본의 첫 단어라도 반환
        if not result and words:
            result = words[0]
                
        return result

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

    def hybrid_search(self, user_ingredients, n_results=5):
        """
        벡터 유사도(60%) + 키워드 매칭(40%) + 자취생용 다양성 필터
        """
        # 로컬 모델용: reindex.py와 동일한 형식으로 쿼리 생성
        ingredients_text = ", ".join(user_ingredients)
        query_text = f"요리명: , 재료: {ingredients_text}"
        query_vector = self.get_embedding(query_text)

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
        
        # 반환 직전 최종 5개에 대해서만 OpenAI LLM 정제 수행
        print("🪄 유튜브 검색 최적화를 위해 요리명을 정제 중입니다...")
        for res in hybrid_results:
            res['name'] = self.clean_with_llm(res['original_name'])

        return hybrid_results

# --- 테스트 및 통합용 출력 코드 ---
if __name__ == "__main__":
    searcher = RecipeSearcher()
    ocr_output = ["감자", "당근", "애호박", "돼지고기"]
    
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