# Демо-данные для кластеризации

## Файл: `ai_data/demo_sample.csv.zip`

Тестовый набор данных из 1000 случайных сэмплов, извлеченных из исходных parquet файлов.

### Характеристики

- **Количество записей**: 1000
- **Формат**: CSV в ZIP архиве
- **Размер**: ~66 KB (архив), ~208 KB (распакованный CSV)
- **Колонки**:
  - `conversation_id` - идентификатор диалога
  - `speaker` - кто говорит (оператор/клиент)
  - `date_time` - время сообщения
  - `text` - текст обращения
  - `request_id` - уникальный идентификатор (формат: `demo-XXXX`)

### Использование

```bash
# Распаковать данные
unzip ai_data/demo_sample.csv.zip -d ai_data/

# Запустить кластеризацию на 20 сэмплов
source venv/bin/activate
export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/demo_sample.csv \
  --limit 20 \
  --batch-id demo_test

# Запустить на 100 сэмплов
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/demo_sample.csv \
  --limit 100 \
  --batch-id demo_100

# Запустить на всех 1000 сэмплов
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/demo_sample.csv \
  --batch-id demo_full
```

### Использование как библиотека

```python
import pandas as pd
from llm_clustering import ClusteringPipeline
import zipfile

# Распаковать и загрузить
with zipfile.ZipFile('ai_data/demo_sample.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('ai_data/')

df = pd.read_csv('ai_data/demo_sample.csv')

# Кластеризация
pipeline = ClusteringPipeline()
result = pipeline.fit(df.head(20), text_column='text')

print(f"Найдено кластеров: {len(result.clusters)}")
print(f"Покрытие: {result.coverage:.1f}%")
```

### Создание собственных демо-данных

Если нужно создать новую выборку:

```python
from llm_clustering.data.subs_dataset import load_subs_messages
import pandas as pd
import zipfile

# Загрузить данные
df = load_subs_messages(sample_size=10000)

# Случайная выборка
demo_df = df.sample(n=1000, random_state=42).reset_index(drop=True)
demo_df['request_id'] = [f"demo-{idx:04d}" for idx in range(len(demo_df))]

# Сохранить
demo_df.to_csv('ai_data/demo_sample.csv', index=False)

# Заархивировать
with zipfile.ZipFile('ai_data/demo_sample.csv.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('ai_data/demo_sample.csv', 'demo_sample.csv')
```

### Назначение

Эти демо-данные предназначены для:
- Быстрого старта и ознакомления с проектом
- Тестирования функциональности без настройки полного датасета
- Демонстрации возможностей кластеризации
- Примеров в документации

Для полноценного использования рекомендуется настроить доступ к полному датасету в `~/data/subs/`.

