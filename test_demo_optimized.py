#!/usr/bin/env python3
"""
Оптимизированный тест на demo_sample.csv с уменьшенным размером батча
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_clustering import ClusteringPipeline, Settings
from llm_clustering.llm import OllamaProvider

print("="*80)
print("ОПТИМИЗИРОВАННЫЙ ТЕСТ: 100 обращений из demo_sample.csv")
print("="*80)
print()

# Загружаем первые 100 записей для теста
DEMO_FILE = PROJECT_ROOT / "ai_data" / "demo_sample.csv"
df = pd.read_csv(DEMO_FILE).head(100)

print(f"✓ Загружено {len(df)} обращений для теста")
print()

# Создаем провайдер с увеличенным timeout
ollama = OllamaProvider()
ollama.temperature = 0.7
ollama.max_tokens = 8000

print(f"Ollama провайдер:")
print(f"  - URL: {ollama.api_url}")
print(f"  - Model: {ollama.model}")
print()

# Оптимизированные настройки - маленькие батчи
settings = Settings(
    clustering_batch_size=10,   # Маленький батч для быстрой обработки
    max_clusters_per_batch=5,   # Меньше кластеров за раз
    min_requests_per_cluster=2,
    default_temperature=0.7,
    parallel_batch_size=3,      # Меньше параллельных запросов
)

# Создаем pipeline
pipeline = ClusteringPipeline(
    llm_provider=ollama,
    settings=settings
)

print("Настройки:")
print(f"  - Размер батча: {settings.clustering_batch_size}")
print(f"  - Макс. кластеров за батч: {settings.max_clusters_per_batch}")
print(f"  - Параллельных запросов: {settings.parallel_batch_size}")
print()

print("="*80)
print("ЗАПУСК КЛАСТЕРИЗАЦИИ")
print("="*80)
print()

start_time = datetime.now()

try:
    batch_count = 0
    for partial_result in pipeline.fit_partial(
        df,
        text_column="text",
        batch_size=settings.clustering_batch_size
    ):
        batch_count += 1
        progress = (partial_result.processed_rows / partial_result.total_rows) * 100
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"Батч {batch_count}:")
        print(f"  ├─ Обработано: {partial_result.processed_rows}/{partial_result.total_rows} ({progress:.1f}%)")
        print(f"  ├─ Новых кластеров: {len(partial_result.new_clusters)}")
        print(f"  ├─ Всего кластеров: {len(pipeline.get_clusters())}")
        print(f"  └─ Время: {elapsed:.1f}с")
        
        if partial_result.new_clusters:
            for cluster in partial_result.new_clusters[:2]:
                print(f"      • {cluster.name}")
        print()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Получаем финальные результаты
    clusters = pipeline.get_clusters()
    
    print("="*80)
    print("РЕЗУЛЬТАТЫ")
    print("="*80)
    print()
    print(f"✓ Кластеризация завершена за {elapsed:.1f} секунд ({elapsed/60:.1f} минут)")
    print(f"  - Обработано: {len(df)} обращений")
    print(f"  - Найдено кластеров: {len(clusters)}")
    print(f"  - Скорость: {len(df)/elapsed*60:.1f} обращений/минуту")
    print()
    
    # Топ-10 кластеров
    print("ТОП-10 КЛАСТЕРОВ:")
    print()
    
    sorted_clusters = sorted(clusters, key=lambda c: c.count, reverse=True)
    for i, cluster in enumerate(sorted_clusters[:10], 1):
        print(f"{i:2d}. {cluster.name} ({cluster.count} обращений)")
    
    print()
    print("✅ ТЕСТ УСПЕШЕН!")
    
except Exception as e:
    print(f"\n✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

