"""
레시피 크롤러
- 1000개 레시피 데이터 수집
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
import random

class RecipeScraper:
    """만개의레시피 크롤러"""
    
    def __init__(self):
        self.base_url = "https://www.10000recipe.com" # 만개의 레시피로
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        self.recipes = []
        self.progress_file = Path("data/progress.json")
        self.output_file = Path("data/raw_recipes.json")
        
        # 디렉토리 생성
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 진행상황 로드
        self.load_progress()
    
    def load_progress(self):
        """진행상황 로드"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.recipes = data.get('recipes', [])
                print(f"✅ Loaded {len(self.recipes)} existing recipes")
        else:
            print("🆕 Starting fresh scraping")
    
    def save_progress(self):
        """진행상황 저장"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'recipes': self.recipes,
                'total': len(self.recipes)
            }, f, ensure_ascii=False, indent=2)
        
        # 최종 결과도 저장
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.recipes, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Progress saved: {len(self.recipes)} recipes")
    
    def get_recipe_list(self, category: str = None, page: int = 1) -> List[str]:
        """
        레시피 목록 페이지에서 URL 수집
        
        Args:
            category: 카테고리 (None이면 전체)
            page: 페이지 번호
            
        Returns:
            레시피 URL 리스트
        """
        try:
            if category:
                url = f"{self.base_url}/recipe/list.html?cat={category}&page={page}"
            else:
                url = f"{self.base_url}/recipe/list.html?page={page}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 레시피 링크 추출
            recipe_links = []
            recipe_items = soup.select('.common_sp_list_ul li')
            
            for item in recipe_items:
                link = item.select_one('a')
                if link and link.get('href'):
                    recipe_url = link['href']
                    if recipe_url.startswith('/recipe/'):
                        recipe_links.append(f"{self.base_url}{recipe_url}")
            
            print(f"📄 Found {len(recipe_links)} recipes on page {page}")
            return recipe_links
            
        except Exception as e:
            print(f"❌ Error getting recipe list: {e}")
            return []
    
    def parse_ingredient(self, ingredient_text: str) -> str:
        """
        재료 텍스트 파싱 (기본 정리)
        
        Args:
            ingredient_text: 원본 재료 텍스트
            
        Returns:
            정리된 재료 텍스트
        """
        # 공백 정리
        ingredient_text = re.sub(r'\s+', ' ', ingredient_text).strip()
        
        # 괄호 안 선택사항 제거 (예: "소금(약간)")
        # ingredient_text = re.sub(r'\([^)]*\)', '', ingredient_text)
        
        return ingredient_text
    
    def scrape_recipe_detail(self, url: str) -> Optional[Dict]:
        """
        개별 레시피 상세 정보 크롤링
        
        Args:
            url: 레시피 URL
            
        Returns:
            레시피 데이터 딕셔너리
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 레시피 ID 추출
            recipe_id = url.split('/')[-1]
            
            # 레시피 이름
            name_elem = soup.select_one('.view2_summary h3')
            name = name_elem.text.strip() if name_elem else "Unknown"
            
            # 재료 추출
            ingredients = []
            ingredient_elems = soup.select('.ready_ingre3 ul li')
            for elem in ingredient_elems:
                # 재료명과 양 추출
                ing_text = elem.get_text(strip=True)
                if ing_text:
                    ingredients.append(self.parse_ingredient(ing_text))
            
            # 조리 순서
            steps = []
            step_elems = soup.select('.view_step_cont')
            for i, elem in enumerate(step_elems, 1):
                step_text = elem.get_text(strip=True)
                if step_text:
                    steps.append(f"{i}. {step_text}")
            
            # 카테고리
            category_elem = soup.select_one('.view2_summary_info .category')
            category = category_elem.text.strip() if category_elem else "기타"
            
            # 난이도, 시간, 인분
            info_elems = soup.select('.view2_summary_info span')
            difficulty = "보통"
            cooking_time = "30분"
            servings = 2
            
            for elem in info_elems:
                text = elem.get_text(strip=True)
                if "난이도" in text or "초급" in text or "중급" in text or "고급" in text:
                    if "초급" in text or "쉽" in text:
                        difficulty = "쉬움"
                    elif "고급" in text or "어려" in text:
                        difficulty = "어려움"
                    else:
                        difficulty = "보통"
                elif "분" in text:
                    cooking_time = text
                elif "인분" in text:
                    servings_match = re.search(r'\d+', text)
                    if servings_match:
                        servings = int(servings_match.group())
            
            # 칼로리 & 영양 정보 (있으면)
            calories = None
            nutrition = {}
            
            nutrition_elem = soup.select_one('.view2_summary_info2')
            if nutrition_elem:
                nutrition_text = nutrition_elem.get_text()
                
                # 칼로리 추출
                cal_match = re.search(r'(\d+)\s*kcal', nutrition_text)
                if cal_match:
                    calories = int(cal_match.group(1))
                
                # 영양소 추출 (나트륨, 단백질 등)
            
            # 설명 
            description_elem = soup.select_one('.view2_summary_in')
            description = description_elem.text.strip() if description_elem else ""
            
            recipe_data = {
                'id': recipe_id,
                'name': name,
                'ingredients': ingredients,
                'steps': steps,
                'category': category,
                'difficulty': difficulty,
                'cooking_time': cooking_time,
                'servings': servings,
                'calories': calories,
                'nutrition': nutrition,
                'description': description,
                'blog_url': url,
                'youtube_url': None,  # 필요하다면...
                'source': '10000recipe'
            }
            
            print(f"✅ Scraped: {name}")
            return recipe_data
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None
    
    def scrape_multiple_pages(self, target_count: int = 1000, start_page: int = 1):
        """
        여러 페이지 크롤링
        
        Args:
            target_count: 목표 레시피 수
            start_page: 시작 페이지
        """
        current_count = len(self.recipes)
        page = start_page
        
        print(f"\n🎯 Target: {target_count} recipes")
        print(f"📊 Current: {current_count} recipes")
        print(f"📄 Starting from page {page}\n")
        
        # 이미 수집한 URL 저장 (중복 방지)
        collected_urls = {r['blog_url'] for r in self.recipes}
        
        while current_count < target_count:
            print(f"\n{'='*60}")
            print(f"📄 Scraping page {page}...")
            print(f"{'='*60}")
            
            # 레시피 목록 가져오기
            recipe_urls = self.get_recipe_list(page=page)
            
            if not recipe_urls:
                print("⚠️  No more recipes found")
                break
            
            # 각 레시피 상세 크롤링
            for url in recipe_urls:
                if current_count >= target_count:
                    break
                
                # 중복 체크
                if url in collected_urls:
                    print(f"⏭️  Skipping duplicate: {url}")
                    continue
                
                # 레시피 크롤링
                recipe_data = self.scrape_recipe_detail(url)
                
                if recipe_data:
                    self.recipes.append(recipe_data)
                    collected_urls.add(url)
                    current_count += 1
                    
                    print(f"📊 Progress: {current_count}/{target_count}")
                    
                    # 10개마다 저장
                    if current_count % 10 == 0:
                        self.save_progress()
                
                # Rate limiting (예의)
                time.sleep(random.uniform(0.5, 1.5))
            
            page += 1
            
            # 페이지 사이에 약간 긴 대기
            time.sleep(random.uniform(2, 4))
        
        # 최종 저장
        self.save_progress()
        
        print(f"\n{'='*60}")
        print(f"🎉 Scraping completed!")
        print(f"📊 Total recipes: {len(self.recipes)}")
        print(f"💾 Saved to: {self.output_file}")
        print(f"{'='*60}")

def main():
    """메인 실행"""
    scraper = RecipeScraper()
    
    # 1000개 레시피 크롤링
    scraper.scrape_multiple_pages(target_count=1000, start_page=1)
    
    print("\n✅ Done!")
    print(f"📁 Output: data/raw_recipes.json")

if __name__ == "__main__":
    main()