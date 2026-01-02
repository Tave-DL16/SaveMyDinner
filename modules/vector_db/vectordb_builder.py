"""
벡터 DB 구축 모듈
정규화된 레시피 데이터를 임베딩하여 Chroma에 저장
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import time

# .env 로드
load_dotenv()

class VectorDBBuilder:
    """벡터 DB 구축기"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        vectordb_path: str = "./modules/vector_db/vectordb_recipes",
        collection_name: str = "recipes_1000"
    ):
        """
        초기화
        
        Args:
            openai_api_key: OpenAI API 키
            vectordb_path: Chroma DB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # OpenAI 클라이언트
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Chroma 클라이언트
        self.vectordb_path = Path(vectordb_path)
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=str(self.vectordb_path))
        
        # 컬렉션
        self.collection_name = collection_name
        self.collection = None
        
        print(f"✅ VectorDBBuilder initialized")
        print(f"   Vector DB: {self.vectordb_path}")
        print(f"   Collection: {self.collection_name}")
    
    def create_embedding_text(self, recipe: Dict) -> str:
        """
        레시피를 임베딩용 텍스트로 변환
        
        전략: 재료 + 요리명 + 설명 통합
        
        Args:
            recipe: 레시피 딕셔너리
            
        Returns:
            임베딩용 텍스트
        """
        # 표준 재료 리스트
        ingredients = recipe.get('ingredients_canonical', recipe.get('ingredients', []))
        ingredients_text = ", ".join(ingredients)
        
        # 카테고리별 재료
        category_info = ""
        if 'ingredients_by_category' in recipe:
            categories = []
            for cat, ings in recipe['ingredients_by_category'].items():
                if ings:
                    categories.append(f"{cat}: {', '.join(ings[:3])}")  # 각 카테고리당 최대 3개
            if categories:
                category_info = " | " + " | ".join(categories)
        
        # 텍스트 조합
        text = (
            f"요리명: {recipe['name']}. "
            f"카테고리: {recipe.get('category', '기타')}. "
            f"재료: {ingredients_text}"
            f"{category_info}. "
            f"설명: {recipe.get('description', '')[:200]}"  # 설명은 200자까지만
        )
        
        return text
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            model: OpenAI 임베딩 모델
            
        Returns:
            임베딩 벡터
        """
        try:
            response = self.openai_client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            return None
    
    def build_vectordb(
        self,
        input_file: str = "data/recipes/normalized_recipes.json",
        batch_size: int = 10,
        force_rebuild: bool = False
    ):
        """
        벡터 DB 구축
        
        Args:
            input_file: 정규화된 레시피 파일
            batch_size: 배치 크기 (한번에 몇 개씩 임베딩)
            force_rebuild: 기존 DB 삭제 후 재구축
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"❌ File not found: {input_file}")
            return False
        
        # 레시피 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"📊 BUILDING VECTOR DATABASE")
        print(f"{'='*60}")
        print(f"Total recipes: {len(recipes)}")
        print(f"Batch size: {batch_size}")
        print(f"Model: text-embedding-3-small")
        print(f"{'='*60}\n")
        
        # 컬렉션 생성/로드
        existing_collections = [c.name for c in self.chroma_client.list_collections()]
        
        if self.collection_name in existing_collections:
            if force_rebuild:
                print(f"🗑️  Deleting existing collection: {self.collection_name}")
                self.chroma_client.delete_collection(self.collection_name)
            else:
                print(f"⚠️  Collection already exists: {self.collection_name}")
                print(f"    Use force_rebuild=True to rebuild")
                self.collection = self.chroma_client.get_collection(self.collection_name)
                return True
        
        # 새 컬렉션 생성
        print(f"🔨 Creating new collection: {self.collection_name}")
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Korean recipes with normalized ingredients"}
        )
        
        # 배치 처리
        total_batches = (len(recipes) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(recipes), batch_size):
            batch = recipes[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            print(f"\n📦 Batch {batch_num}/{total_batches}")
            print(f"   Processing recipes {batch_idx + 1} to {min(batch_idx + batch_size, len(recipes))}")
            
            # 임베딩 생성
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for recipe in batch:
                # 임베딩용 텍스트 생성
                text = self.create_embedding_text(recipe)
                documents.append(text)
                
                # 임베딩 생성
                embedding = self.get_embedding(text)
                if embedding:
                    embeddings.append(embedding)
                else:
                    # 실패 시 빈 벡터 (차원 맞추기)
                    embeddings.append([0.0] * 1536)
                
                # 메타데이터 준비
                metadata = {
                    'name': recipe['name'],
                    'category': recipe.get('category', '기타'),
                    'difficulty': recipe.get('difficulty', '보통'),
                    'cooking_time': recipe.get('cooking_time', '알 수 없음'),
                    'servings': recipe.get('servings', 2),
                    'ingredients': json.dumps(recipe.get('ingredients_canonical', []), ensure_ascii=False),
                    'calories': recipe.get('calories') if recipe.get('calories') else 0,
                }
                
                # 블로그 URL (있으면)
                if recipe.get('blog_url'):
                    metadata['blog_url'] = recipe['blog_url']
                
                metadatas.append(metadata)
                
                # ID (레시피 ID 또는 인덱스)
                recipe_id = recipe.get('id', f"recipe_{batch_idx + len(ids)}")
                ids.append(str(recipe_id))
            
            # Chroma에 저장
            try:
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"   ✅ Batch {batch_num} saved")
            except Exception as e:
                print(f"   ❌ Error saving batch {batch_num}: {e}")
            
            # Rate limiting (API 호출 제한 고려)
            time.sleep(0.5)
        
        print(f"\n{'='*60}")
        print(f"🎉 VECTOR DB BUILD COMPLETED!")
        print(f"{'='*60}")
        print(f"Total recipes: {len(recipes)}")
        print(f"Collection: {self.collection_name}")
        print(f"Path: {self.vectordb_path}")
        print(f"{'='*60}\n")
        
        return True
    
    def get_stats(self) -> Dict:
        """벡터 DB 통계"""
        if not self.collection:
            # 컬렉션 로드 시도
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
            except:
                return {'error': 'Collection not found'}
        
        count = self.collection.count()
        return {
            'total_recipes': count,
            'collection_name': self.collection_name,
            'vectordb_path': str(self.vectordb_path)
        }
    
    def test_search(self, query: str, n_results: int = 5):
        """
        검색 테스트
        
        Args:
            query: 검색 쿼리
            n_results: 결과 수
        """
        if not self.collection:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        
        print(f"\n🔍 Testing search: '{query}'")
        
        # 쿼리 임베딩
        query_embedding = self.get_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to create query embedding")
            return
        
        # 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        print(f"\n📊 Top {n_results} results:\n")
        
        for i in range(len(results['ids'][0])):
            recipe_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            similarity = round((1 - distance) * 100, 1)
            metadata = results['metadatas'][0][i]
            
            print(f"{i+1}. {metadata['name']} (유사도: {similarity}%)")
            print(f"   카테고리: {metadata['category']}")
            print(f"   난이도: {metadata['difficulty']} | 시간: {metadata['cooking_time']}")
            
            ingredients = json.loads(metadata['ingredients'])
            print(f"   재료: {', '.join(ingredients[:5])}...")
            print()

def main():
    """메인 실행"""
    builder = VectorDBBuilder()
    
    # 벡터 DB 구축
    print("\n🚀 Starting Vector DB build...")
    builder.build_vectordb(
        input_file="data/recipes/normalized_recipes.json",
        batch_size=10,
        force_rebuild=False
    )
    
    # 통계
    stats = builder.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Total recipes in DB: {stats['total_recipes']}")
    
    # 테스트 검색
    print("\n" + "="*60)
    print("🧪 TESTING SEARCHES")
    print("="*60)
    
    # 테스트 1: 재료 기반
    builder.test_search("돼지고기, 김치, 두부", n_results=3)
    
    # 테스트 2: 요리 종류
    builder.test_search("간단하고 빠른 볶음 요리", n_results=3)
    
    # 테스트 3: 해물
    builder.test_search("새우, 오징어", n_results=3)
    
    print("\n✅ Done!")
    print(f"📁 Vector DB saved to: {builder.vectordb_path}")

if __name__ == "__main__":
    main()