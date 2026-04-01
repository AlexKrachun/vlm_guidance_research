
# Репозиторий к исследованию возможностей vlm guidance для диффузионной генерации и редактирования изображений

Доклад о содержании исследования: [pdf](./report.pdf)

Автор исполнял код на Python 3.10.20, cuda 12.0, 12.6, 12.8 на A100 80 gb vram. 

Для запуска vlm guidance пайплайна необходимо 39 gb vram. Для всех остальных пайплайнов этого тоже хватит


## Установка репозитория и создание и окружения.

```shell
git clone git@github.com:AlexKrachun/vlm_guidance_research_dev.git
cd vlm_guidance_research_dev/
conda env create -f environment.yaml
conda activate t2v
pip install flash-attn --no-build-isolation
pip install hydra-core

```

## Запуск трех приведенных пайплайнов генерации
Для исполнения flux пайплайна надо авторизоваться в hugging face:

```shell
hf auth login
```



Запустить vanilla sd1.5, vqa guided sd1.5, flux1-dev на своем промпте
```shell
python -m vlm_guidance_project.vlm_guidance.run \
  run.prompt="a cat on a chair" \
  run.vqa_score=True \
  run.vanilla_sd=True \
  run.flux1=True
```


Запустить vanilla sd1.5, vqa guided sd1.5, flux1-dev на текстовом файле datasets/subset.txt - где каждая строка - это один промпт (можете вписать свои промпты)
```shell
python -m vlm_guidance_project.vlm_guidance.batch_run \
  batch.prompts_file=../datasets/subset.txt \
  run.vqa_score=True \
  run.vanilla_sd=True \
  run.flux1=True \
  batch.output_root_dir=subset_generations    
```


## Подсчет и визуализация метрик

### CLIP score

Посчитат и сохранить в csv файл clip score по папке с изображениями полученными разными пайплайнами (папка получена с помощью прогона `python -m vlm_guidance.batch_run`)
```shell
python3 metrics/clip_score_clalc.py \
  -generations vlm_guidance_project/subset_generations \
  --output metrics/clip_score_subset_result.csv
```

Построить графики clip score по csv со значениями метрики 
```shell
python3 metrics/clip_visualize.py \
--input metrics/clip_score_simple_result.csv \
--output-dir metrics/clip_plots_simple_plots
```


### Alignment и Quality scores
Для подсчета метрик alignment и quality
```shell
export OPENAI_API_KEY="YOUR OPENAI API KEY"
```

```shell
python3 metrics/alignment_score_clalc.py \
  -generations vlm_guidance_project/subset_generations \
  --output metrics/alignment_score_subset_result.csv \
  --api-key "$OPENAI_API_KEY" \
  --concurrency 10
```

Построить графики alignment и quality score по csv со значениями метрик
```shell
python3 metrics/alignment_visualize.py \
  --input metrics/alignment_score_subset_result.csv \
  --output-dir metrics/alignment_simple_plots
```



### Визуализация статистик хода генерации

Сгенерировать граяфик статистики градиентов vlm гайденса
```shell
python metrics/statistics.py --input-root vlm_guidance_project/subset_generations
```



## Запуск пайплайнов редактирования

запустить null-text editing на датасете datasets/coco/first3
```shell
python -m vlm_guidance_editing.vlm_guidance.run \
  run.dataset_root=datasets/coco/first3 \
  run.pipeline_null_text_inversion=True \
  run.pipeline_vlm_guided_editing=True \
  run.output_root_dir=vlm_guidance_editing/first3_resluts \
  algorithm.save_debug_tensors=False \
  algorithm.gd_only_first_k_steps=15 \
  guided.gd_steps=3 \
  guided.zt_optimizing=true \
  guided.zt_lr=1 \
  guided.null_text_emb_optimizing=true \
  guided.null_text_emb_lr=1

```



