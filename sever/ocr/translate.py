from deep_translator import GoogleTranslator

def translate_ingredients(
    english_list
):
    """
    Translates a list of English ingredient names to Korean using Google Translate API (via deep-translator).
    """
    if not english_list:
        return []

    korean_list = []
    print(f"Translating {len(english_list)} ingredients using Google Translate...")

    try:
        translator = GoogleTranslator(source='en', target='ko')
        
        for item in english_list:
            if not item or not item.strip():
                continue
            
            # Translate item
            translated_name = translator.translate(item.strip())
            
            if translated_name:
                print(f"  Translated: {item} -> {translated_name}")
                korean_list.append(translated_name)
                
    except Exception as e:
        print(f"Google Translate error: {e}")
        # Return partial list if some were translated, or empty list
    
    return list(dict.fromkeys(korean_list))
