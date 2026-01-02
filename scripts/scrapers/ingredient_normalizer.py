"""
재료 정규화 모듈
3단계 정규화 전략: 기본 정규화 → 동의어 매핑 → 카테고리 분류
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

class IngredientNormalizer:
    """재료 정규화기"""
    
    def __init__(self):
        """초기화 - 동의어 사전 및 카테고리 정의"""
        
        # ============ Level 2: 동의어 사전 ============
        self.SYNONYMS = {
            # 파류
            "대파": ["파", "쪽파", "실파"],
            "파": ["대파", "쪽파", "실파"],
            
            # 고기류
            "돼지고기": ["삼겹살", "목살", "등심", "앞다리살", "뒷다리살"],
            "소고기": ["우둔", "사태", "양지", "등심", "목심", "안심"],
            "닭고기": ["닭", "닭가슴살", "닭다리살", "닭날개"],
            
            # 마늘
            "마늘": ["다진마늘", "편마늘", "마늘편"],
            
            # 생강
            "생강": ["생강즙", "다진생강"],
            
            # 고추
            "고추": ["청양고추", "홍고추", "풋고추"],
            "고춧가루": ["고추가루"],
            
            # 간장
            "간장": ["양조간장", "진간장", "국간장"],
            
            # 된장
            "된장": ["쌈장", "막된장"],
            
            # 설탕
            "설탕": ["백설탕", "황설탕"],
            
            # 식초
            "식초": ["현미식초", "사과식초"],
            
            # 참기름
            "참기름": ["참깨기름"],
            
            # 김치
            "김치": ["배추김치", "포기김치"],
            
            # 두부
            "두부": ["연두부", "순두부"],
        }
        
        # ============ Level 3: 카테고리 분류 ============
        self.CATEGORIES = {
            "육류": [
                "소고기", "돼지고기", "닭고기", "오리고기", "양고기",
                "삼겹살", "목살", "등심", "안심", "우둔", "사태", "양지",
                "닭가슴살", "닭다리", "다짐육", "간", "곱창"
            ],
            "해산물": [
                "새우", "오징어", "낙지", "문어", "조개", "굴", "전복",
                "고등어", "삼치", "갈치", "명태", "연어", "참치", "멸치",
                "바지락", "홍합", "게", "꽃게"
            ],
            "채소": [
                "양파", "당근", "감자", "고구마", "무", "배추", "양배추",
                "브로콜리", "시금치", "상추", "깻잎", "호박", "애호박",
                "가지", "파프리카", "피망", "오이", "토마토", "버섯",
                "느타리버섯", "팽이버섯", "표고버섯", "양송이버섯"
            ],
            "파류": [
                "대파", "파", "쪽파", "실파", "부추"
            ],
            "고추마늘": [
                "마늘", "다진마늘", "편마늘", "생강", "다진생강",
                "고추", "청양고추", "홍고추", "풋고추", "고춧가루"
            ],
            "조미료": [
                "소금", "설탕", "간장", "된장", "고추장", "쌈장",
                "식초", "참기름", "들기름", "올리브오일", "식용유",
                "후추", "깨소금", "통깨", "참깨", "맛술", "청주",
                "굴소스", "두반장", "케첩", "마요네즈", "머스타드"
            ],
            "곡물": [
                "쌀", "밥", "찹쌀", "현미", "보리", "밀가루", "부침가루",
                "튀김가루", "빵가루", "면", "우동면", "당면", "라면"
            ],
            "유제품": [
                "우유", "생크림", "치즈", "모짜렐라", "파마산", "버터", "요거트"
            ],
            "콩제품": [
                "두부", "순두부", "연두부", "콩나물", "숙주"
            ],
            "김치류": [
                "김치", "배추김치", "깍두기", "총각김치"
            ],
            "기타": []
        }
        
        # 역매핑: 재료 → 카테고리
        self.ingredient_to_category = {}
        for category, ingredients in self.CATEGORIES.items():
            for ing in ingredients:
                self.ingredient_to_category[ing] = category
    
    # ============ Level 1: 기본 정규화 ============
    
    def normalize_basic(self, ingredient: str) -> str:
        """
        기본 정규화
        
        Args:
            ingredient: 원본 재료 텍스트
            
        Returns:
            정규화된 재료명
        """
        # 1. 공백 정리
        normalized = re.sub(r'\s+', '', ingredient)
        
        # 2. 숫자 + 단위 제거
        # 예: "돼지고기 300g" → "돼지고기"
        # 예: "양파 1개" → "양파"
        # 예: "간장 2큰술" → "간장"
        normalized = re.sub(r'\d+\.?\d*\s*(g|kg|ml|l|cc|개|조각|큰술|작은술|T|t|컵|스푼|봉|팩|캔)', '', normalized, flags=re.IGNORECASE)
        
        # 2-1. 불필요한 텍스트 제거 (구매, 준비 등)
        normalized = re.sub(r'(구매|준비|필요)', '', normalized)
        
        # 3. 괄호 제거
        # 예: "소금(약간)" → "소금"
        # 일단 괄호는 제거하지 않고 나중에 처리
        
        # 4. 특수문자 제거 (일부)
        normalized = re.sub(r'[^\w가-힣()]', '', normalized)
        
        # 5. 소문자 변환 (영어만)
        normalized = normalized.lower() 
        
        return normalized.strip()
    
    def extract_ingredient_name(self, ingredient: str) -> Tuple[str, str]:
        """
        재료명과 상세 정보 분리
        
        Args:
            ingredient: "돼지고기 300g" 또는 "양파(중) 1개"
            
        Returns:
            (재료명, 전체 텍스트)
        """
        # 기본 정규화
        normalized = self.normalize_basic(ingredient)
        
        # 순수 재료명 추출 (한글 + 영문만)
        ingredient_name = re.sub(r'[0-9()]', '', normalized)
        ingredient_name = ingredient_name.strip()
        
        return ingredient_name, ingredient
    
    # ============ Level 2: 동의어 매핑 ============
    
    def find_canonical_form(self, ingredient: str) -> str:
        """
        동의어를 표준형으로 변환
        
        Args:
            ingredient: 정규화된 재료명
            
        Returns:
            표준 재료명
        """
        # 이미 표준형인 경우
        if ingredient in self.SYNONYMS:
            return ingredient
        
        # 동의어 검색
        for canonical, synonyms in self.SYNONYMS.items():
            if ingredient in synonyms:
                return canonical
        
        # 부분 매칭 (포함 관계)
        for canonical, synonyms in self.SYNONYMS.items():
            # "닭가슴살" → "닭고기"
            if any(syn in ingredient for syn in synonyms):
                return canonical
            if any(ingredient in syn for syn in synonyms):
                return canonical
        
        return ingredient
    
    # ============ Level 3: 카테고리 분류 ============
    
    def categorize_ingredient(self, ingredient: str) -> str:
        """
        재료 카테고리 분류
        
        Args:
            ingredient: 표준 재료명
            
        Returns:
            카테고리명
        """
        # 정확히 매칭되는 카테고리
        if ingredient in self.ingredient_to_category:
            return self.ingredient_to_category[ingredient]
        
        # 부분 매칭
        for category, ingredients in self.CATEGORIES.items():
            for ing in ingredients:
                if ing in ingredient or ingredient in ing:
                    return category
        
        return "기타"
    
    # ============ 통합 정규화 ============
    
    def normalize_ingredient(self, ingredient: str) -> Dict:
        """
        3단계 정규화 통합
        
        Args:
            ingredient: 원본 재료 텍스트
            
        Returns:
            {
                'original': 원본,
                'normalized': 정규화된 이름,
                'canonical': 표준형,
                'category': 카테고리
            }
        """
        # Level 1: 기본 정규화
        ingredient_name, full_text = self.extract_ingredient_name(ingredient)
        
        # Level 2: 동의어 매핑
        canonical = self.find_canonical_form(ingredient_name)
        
        # Level 3: 카테고리 분류
        category = self.categorize_ingredient(canonical)
        
        return {
            'original': ingredient,
            'normalized': ingredient_name,
            'canonical': canonical,
            'category': category
        }
    
    def normalize_recipe_ingredients(self, ingredients: List[str]) -> List[Dict]:
        """
        레시피의 모든 재료 정규화
        
        Args:
            ingredients: 재료 리스트
            
        Returns:
            정규화된 재료 리스트
        """
        return [self.normalize_ingredient(ing) for ing in ingredients]
    
    # ============ 데이터 처리 ============
    
    def process_recipes(
        self, 
        input_file: str = "data/raw_recipes.json",
        output_file: str = "data/normalized_recipes.json"
    ):
        """
        전체 레시피 데이터 정규화
        
        Args:
            input_file: 입력 파일
            output_file: 출력 파일
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            print(f"❌ File not found: {input_file}")
            return
        
        # 레시피 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        print(f"✅ Loaded {len(recipes)} recipes")
        print(f"🔄 Normalizing ingredients...\n")
        
        # 재료 통계
        all_ingredients = set()
        category_stats = defaultdict(int)
        
        # 각 레시피 처리
        for i, recipe in enumerate(recipes, 1):
            # 원본 재료 보존
            recipe['ingredients_raw'] = recipe['ingredients'].copy()
            
            # 재료 정규화
            normalized_ingredients = self.normalize_recipe_ingredients(recipe['ingredients'])
            
            # 정규화 결과 저장
            recipe['ingredients_normalized'] = normalized_ingredients
            
            # 간단한 재료 리스트 (canonical만)
            recipe['ingredients_canonical'] = [ing['canonical'] for ing in normalized_ingredients]
            
            # 카테고리별 재료
            recipe['ingredients_by_category'] = defaultdict(list)
            for ing in normalized_ingredients:
                recipe['ingredients_by_category'][ing['category']].append(ing['canonical'])
            
            # 통계 수집
            for ing in normalized_ingredients:
                all_ingredients.add(ing['canonical'])
                category_stats[ing['category']] += 1
            
            # 진행 상황
            if i % 100 == 0:
                print(f"📊 Processed {i}/{len(recipes)} recipes...")
        
        # 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Normalization completed!")
        print(f"💾 Saved to: {output_path}")
        
        # 통계 출력
        print(f"\n{'='*60}")
        print(f"📊 STATISTICS")
        print(f"{'='*60}")
        print(f"Total unique ingredients: {len(all_ingredients)}")
        print(f"\nIngredients by category:")
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category:15s}: {count:4d}")
        
        # 샘플 출력
        print(f"\n{'='*60}")
        print(f"📝 SAMPLE (First recipe)")
        print(f"{'='*60}")
        if recipes:
            sample = recipes[0]
            print(f"Recipe: {sample['name']}")
            print(f"\nOriginal ingredients:")
            for ing in sample['ingredients_raw'][:5]:
                print(f"  - {ing}")
            print(f"\nNormalized:")
            for ing in sample['ingredients_normalized'][:5]:
                print(f"  - {ing['original']:30s} → {ing['canonical']:15s} ({ing['category']})")

def main():
    """메인 실행"""
    normalizer = IngredientNormalizer()
    
    # 레시피 정규화
    normalizer.process_recipes(
        input_file="data/raw_recipes.json",
        output_file="data/normalized_recipes.json"
    )
    
    print("\n✅ Done!")
    print(f"📁 Output: data/normalized_recipes.json")

if __name__ == "__main__":
    main()