
# Репозиторий к исследованию возможностей vlm guidance

Доклад о содержании исследования: [pdf](./vlm_guide_result_report.pdf)

Автор исполнял код на Python 3.10.20, cuda 12.0, 12.6, 12.8 на A100 80 gb vram. 

Для запуска vlm guidance пайплайна необходимо 36 gb vram


## Установка репозитория и создание и окружения.

```shell
git clone https://github.com/AlexKrachun/vlm_guidance_research
cd vlm_guidance_research/
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



