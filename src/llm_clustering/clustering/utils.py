"""Utility functions for clustering module."""

import json
import re
from typing import Any


def extract_json_from_response(response_text: str) -> dict[str, Any]:
    """Extract JSON from LLM response text.
    
    Some models (like Qwen3) return responses with thinking tags like <think>...</think>.
    This function removes those tags and extracts the JSON payload.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed JSON dict
        
    Raises:
        json.JSONDecodeError: If no valid JSON found in response
    """
    # Удаляем теги <think>...</think> и другие подобные
    cleaned_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    cleaned_text = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned_text, flags=re.DOTALL).strip()
    
    # Если после очистки пусто, пробуем извлечь JSON из оригинального текста
    if not cleaned_text:
        cleaned_text = response_text
    
    # Ищем JSON в тексте (между { и } - самый внешний JSON объект)
    # Используем более надежный метод поиска сбалансированных скобок
    start_idx = cleaned_text.find('{')
    if start_idx == -1:
        raise json.JSONDecodeError("No JSON object found", cleaned_text, 0)
    
    # Считаем скобки для нахождения конца JSON объекта
    bracket_count = 0
    in_string = False
    escape_next = False
    end_idx = start_idx
    
    for i in range(start_idx, len(cleaned_text)):
        char = cleaned_text[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
    
    if bracket_count != 0:
        # Fallback к простому regex если подсчет скобок не сработал
        json_match = re.search(r'\{.*\}', cleaned_text, flags=re.DOTALL)
        if json_match:
            cleaned_text = json_match.group(0)
    else:
        cleaned_text = cleaned_text[start_idx:end_idx]
    
    return json.loads(cleaned_text)

