# Исследование сжатия промежуточных CXR-признаков (research-only)

Этот модуль добавлен рядом с существующими `current_project_line.ipynb`,
`v2 copy (1).ipynb`, `BLIP_3_Model_Comparison_270_30.ipynb`, отчётом
`IU_pilot_results.md` и исходными IU X-Ray изображениями. Ничего из них
не нужно удалять, переписывать или перетренировывать для установки модуля.

## Что соответствует трём техническим заданиям

| Документ | Добавленная реализация | Статус/ограничения |
|---|---|---|
| 01. Diagnostic Risk-Constrained | `models.py`, `codec.py`, `losses.py`, `experiment.py` | 1×1 learned bottleneck, STE, weighted/masked BCE, многометочная Bernoulli-дистилляция, ROI-loss для training only, метрики; не доказано клиническое улучшение |
| 02. Uncertainty-Guided Allocation | `allocation.py`, `experiment.py` | score-heads, градиентные teacher-прокси, greedy 2/4/8, истинный размер вместе с allocation map, ablations; entropy-gradient proxy НЕ калиброванная эпистемическая неопределённость |
| 03. Single-Model Variable Rate | `codec.py`, `allocation.py`, `experiment.py`, `deployment.py` | FiLM, случайный bitrate при обучении, битстрим с версией/формой/split/rate ID, 5/10/25/50%-targets, оценка доступности бюджета; DEFLATE вместо обученного ANS/range-coder |

Важно: DOCX — это гипотезы и этапы будущих экспериментов, а не готовый
код или утверждение о достигнутой медицинской точности. В модуле реализована
техническая основа для трёх гипотез. Публикационные эксперименты,
калибровка, внешняя валидация и экспертная оценка остаются задачами проекта.

## Установка

В корне своего существующего репозитория (не вместо него) скопируйте
файлы из ZIP с сохранением вложенных папок. Не копируйте папку ZIP как
новый подпроект: `research_compression/` должна оказаться рядом с README.md.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-research.txt
python -m pytest -q tests
```

Используйте установку `torch`/`torchvision`, совместимую с вашей видеокартой
и CUDA. Если их уже предоставляет рабочее окружение `lamofc`, не заменяйте
бездумно — лучше создайте отдельную среду.

## Сначала подготовьте CSV с метками заболеваний, а не текстами отчётов

Пример `data/cxr_manifest.csv` (гипотетические строки, НЕ датасет):

```csv
patient_id,path,split,Cardiomegaly,Pleural Effusion,Pneumothorax,roi_boxes
patient_001,patient_001/a.png,train,0,1,0,"[[0.15,0.3,0.32,0.5]]"
patient_002,patient_002/a.png,val,0,0,1,
patient_003,patient_003/a.png,test,1,0,-1,
```

- `patient_id` обязательно; один пациент **не может** встречаться в двух split.
- `path` — путь к картинке относительно `image_root`.
- `split` — `train`, `val` или `test`. Для обучения и финальной оценки нужны все три.
- Колонки заболеваний — `0`, `1`, `-1` (uncertain) или пустое значение.
- `uncertainty_policy = ignore|zero|one` обязательно указывается явно.
- `roi_boxes` (опционально) — JSON со списком нормализованных `[x0,y0,x1,y1]`;
  **загружается только на `train`**, ни разу не используется при test-инференсе.
- Нельзя автоматически преобразовывать `y_report` из IU-отчётов в метки
  заболеваний и трактовать это как достоверную разметку: нужен отдельный
  проверенный источник меток (например, CheXpert/MIMIC-CXR после оформления доступа).
- Снимки преобразуются в grayscale, дублируются до трёх каналов,
  масштабируются до `image_size × image_size`, нормализуются ImageNet mean/std.
  Зафиксируйте эту предобработку во всех сравнениях.

В `configs/cxr_research.example.json` пропишите реально существующие
пути, названия колонок заболеваний, `device` и нужную конфигурацию.
Это **пример**, который не запустится без подготовки датасета.

## Последовательность воспроизводимого прогона

Одной командой (аудит, baseline, probe, codec, evaluate, reproducibility.json):

```bash
python -m research_compression.run_all --config configs/cxr_research.example.json
```

Или по стадиям для проверки результатов каждого этапа:

```bash
python -m research_compression.experiment audit --config configs/cxr_research.example.json
python -m research_compression.experiment train-baseline --config configs/cxr_research.example.json
python -m research_compression.experiment probe --config configs/cxr_research.example.json
python -m research_compression.experiment train-codec --config configs/cxr_research.example.json
python -m research_compression.experiment evaluate --config configs/cxr_research.example.json
```

1. `audit` проверяет patient leakage, неопределённые/пропущенные метки,
   распределение split, SHA-256 манифеста.
2. `train-baseline` обучает DenseNet-121 с тремя выбранными признаками,
   сохраняет лучший по validation loss checkpoint. Использовать
   ImageNet-pretrained `pretrained_imagenet=true` можно при доступном
   checkpoint/download. `false` — случайная инициализация, не готовый диагноз.
   При дисбалансе классов можно включить `use_class_weights=true`: веса
   оцениваются ТОЛЬКО по training labels, без подсматривания в val/test.
3. `probe` сериализует **реальные байты** признаков в 2/4/6/8 бит/символ
   и сохраняет MSE/bytes. Это проба на `val`, не финальный test.
4. `train-codec` фиксирует веса обученной DenseNet и обучает FiLM-кодек
   плюс spatial score-heads. Каждая training mini-batch получает одну из
   `rate_fractions`; чекпойнт связан с SHA-256 baseline.
5. `evaluate` выбирает пороги чувствительности по `val`, затем один раз
   считает метрики на test, patient-bootstrap 95% CI, парную разность AUROC
   относительно uncompressed reference и фактические байты/битстримы.
   Проверяются `uniform`, `random`, `saliency`, `uncertainty`, `combined`.
   CSV с предсказаниями и байтами лежат в `output_dir`.

Для разных split-слоёв `denseblock1`, `denseblock2`, `denseblock3` меняйте
`split` и используйте **разные output_dir**: checkpoints несовместимы.
Для трёх random seeds повторяйте полный pipeline с отдельными папками,
фиксируйте seed в конфиге и агрегируйте mean ± sd. Это НЕ автоматизированный
тройной прогон. Для независимых fixed-rate моделей используйте отдельный
тренировочный запуск с одним значением `rate_fractions` и одинаковым budget
compute; сравнивайте с single-model в точках одинакового **achieved rate**.

### Что именно считается «размером»

`encode_symbols()` возвращает реальный бинарный MFCR-поток:
`magic + длина header + JSON header + двумерная allocation map + zlib payload`.
Число переданных бит на снимок — `len(message) * 8`. Заголовки,
пространственная карта и padding **учтены**. Веса модели/codec заранее
известны обеим сторонам; TCP/IP/TLS overhead не входит в это число.

`estimated_bits` в обучении — **суррогатная дифференцируемая оценка**
через независимую Laplace model, а не фактическая длина битстрима.
`budget_met` в результатах показывает реальное соответствие цели.
Greedy allocator находит допустимое решение только по приблизительным
сырым символам и reserve для заголовка: для строгих режимов нужна
последующая калибровка по сериализованным байтам. Даже минимальные 2 бита
могут не помещаться в target — это должно выводиться как `budget_met=false`.

В `codec.py` используется стандартный `zlib` (DEFLATE с Huffman),
**не обученный арифметический/ANS-кодер**. Для научного сравнения
может понадобиться CompressAI/ANS с обученным entropy model и
точным модельным rate, а также версионированный streaming transport.

## Как подключить к существующим R2Gen / BLIP ноутбукам

Реальное имя внутреннего слоя зависит от checkpoint/model: сперва
выведите `module_names(model)`. Не угадывайте `layer4` и не заменяйте
чужой checkpoint случайным кодеком. Требуется **специально обученный
на этом же feature-layer** `codec` и `scorer`; DenseNet checkpoint
нельзя напрямую применять к признакам R2Gen/BLIP с другими каналами.

```python
from research_compression.bridge import (
    module_names, FeatureReplacement, make_replacement_callback
)
# model = ...                 # существующая pretrained R2Gen/BLIP модель
# codec = ...                 # обученный на ТОЧНО тех же feature channels/split
# scorer = ...                # обученный на этих же features
# print(module_names(model))  # выбрать существующий tensor-output слой [B,C,H,W]
# layer = "<real/module/name>"
# records = []
# callback = make_replacement_callback(codec.eval(), scorer.eval(),
#     split=layer, rate_fraction=0.25, mode="combined", rate_log=records)
# with FeatureReplacement(model.eval(), layer, callback):
#     output = model(...)
# print(records)  # actual_bytes, actual_bits, target_bits, budget_met
```

`FeatureReplacement` удаляет hook даже при исключении; проверяет тип/размер
feature, не изменяет файлы исходных ноутбуков, работает **только inference**.
Этот путь даёт техническую интеграцию с существующей генерацией отчётов,
НО не подменяет требуемые DOCX disease AUROC и lesion localization.
При BLIP-токенах `[B,N,D]` нужен отдельный token adapter; текущий hook
поддерживает только пространственные `[B,C,H,W]`.

## Deployment-оценка

`deployment.simulate_latency()` считает в миллисекундах
`T_edge + T_encode + 1000 × actual_bits / bandwidth_bits_per_sec + T_decode + T_cloud`.
`latency_summary()` вычисляет p50/p95. Укажите свои измеренные
edge/cloud времени, сетевую полосу и overhead отдельно; не выдавайте
синтетическое время за аппаратно измеренное. GPU power и peak RAM
текущим модулем отдельно не измеряются.

## Чего пока нет и что ОБЯЗАТЕЛЬНО нужно до MICCAI-статьи

- Надёжной clinical-метрики small-lesion sensitivity по внешним box/mask
  аннотациям; отдельного внешнего dataset-shift анализа.
- Корректно калиброванной эпистемической/алеаторической неопределённости
  (предоставлены только непроверенные градиентные teacher proxy).
- Полноценного learned entropy coder/range coding с training-calibrated PMF;
  точного реального bitrate constraint прямо в optimizer.
- PCA baseline, автоматического fixed-rate sweep, независимого протокола
  end-to-end latency на edge/cloud, анализа power/peak RAM.
- Автоматической генерации ROC/Pareto/lesion heatmap фигур,
  corrected multiple-endpoint inference и Clinical Expert Review.
- Валидации на реальном медицинском датасете; представленные тесты
  проверяют код и форматы, а не качество диагнозов.

**Нельзя заявлять:** «метод улучшил AUROC», «медицински безопасен» или
«научно доказано снижение diagnostic risk» по одним unit tests.
Система предназначена только для исследований, не для решений о пациентах.
