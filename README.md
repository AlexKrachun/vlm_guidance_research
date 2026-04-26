
# Исследование VLM guidance для генерации и редактирования изображений

<p align="center">
  <img src="assets/t_0000_cumulative_before_denoise.png" width="10%"><img src="assets/t_0003_cumulative_before_denoise.png" width="10%"><img src="assets/t_0006_cumulative_before_denoise.png" width="10%"><img src="assets/t_0009_cumulative_before_denoise.png" width="10%"><img src="assets/t_0012_cumulative_before_denoise.png" width="10%"><img src="assets/t_0015_cumulative_before_denoise.png" width="10%"><img src="assets/t_0018_cumulative_before_denoise.png" width="10%"><img src="assets/t_0021_cumulative_before_denoise.png" width="10%"><img src="assets/t_0027_cumulative_before_denoise.png" width="10%">
  <br>
  <img src="assets/t_0000_before_denoise.png" width="10%"><img src="assets/t_0003_before_denoise.png" width="10%"><img src="assets/t_0006_before_denoise.png" width="10%"><img src="assets/t_0009_before_denoise.png" width="10%"><img src="assets/t_0012_before_denoise.png" width="10%"><img src="assets/t_0015_before_denoise.png" width="10%"><img src="assets/t_0018_before_denoise.png" width="10%"><img src="assets/t_0021_before_denoise.png" width="10%"><img src="assets/t_0027_before_denoise.png" width="10%">
</p>

Этот репозиторий содержит код и артефакты исследования, посвященного использованию `vision-language models` для управления диффузионной генерацией и редактированием изображений. Основная идея работы состоит в том, чтобы во время денойзинга использовать сигнал от VLM как функцию согласованности между текущим шумным латентом и промптом. По градиенту этой функции согласованности можно подталкивать шумные латенты в сторону описанного в промпте. Этим мы и занимаемся в этой работе.

Если в двух словах, это аналог classifier guidance, но роль классификатора здесь играет большая авторегрессионная мультимодальная языковая модель - VLM, например Qwen. Мы показываем ей генерируемую картинку и спрашиваем, насколько она соответствует промпту. Далее, по аналогии с подсчетм градиента классификатора, считаем градиент вероятности токена "Yes" и по этому градиенту оптимизируем латенты.

Мы реализовали два сценария:

1. `VLM-guided generation`: сравнение трех пайплайнов text-to-image генерации:
   - `vqa_score` - Stable Diffusion 1.5 с VLM-guidance.
   - `vanilla_sd` - обычный Stable Diffusion 1.5 - бейзлайн.
   - `flux1` - базовый `FLUX.1-dev` как сильный reference бейзлайн.
2. `VLM-guided editing`: редактирование изображений через `null-text inversion` с  VLM-guidance.

Работа отвечает на два вопроса:

1. Улучшает ли VLM-guidance семантическое соответствие изображения тексту по сравнению с базовыми диффузионными моделями.
2. Можно ли использовать тот же принцип не только для генерации, но и для редактирования изображений, особенно на сложных или контекстно нетипичных запросах.

Полный разбор постановки задачи, метода и результатов находится в [report.pdf](./report.pdf).

Ход развития исследования и все подробности можно наблюдать в отчетах об итерациях работы над проектом: [report-1](vlm_guide_result_1_report.pdf), [report-2](vlm_guide_result_2_report.pdf), [report-3](vlm_guide_result_3_report.pdf).

## Что есть в репозитории

- `vlm_guidance_project/` - код пайплайнов генерации: single prompt и batch (запуск на датасете) режимы.
- `vlm_guidance_editing/` - код пайплайнов редактирования через `null-text inversion` и VLM-guided editing.
- `datasets/` - текстовые наборы промптов и примеры входных данных для редактирования.
- `metrics/` - скрипты для подсчета `CLIP score`, `alignment`, `quality` и визуализации результатов.
- `report.pdf` - полный текст исследовательской работы.
- [report-1](vlm_guide_result_1_report.pdf), [report-2](vlm_guide_result_2_report.pdf), [report-3](vlm_guide_result_3_report.pdf) - подробные отчеты об итерациях работы над проектом.

## Что запускать для ознакомления

Если нужно быстро понять проект на практике, достаточно следующей последовательности (команды приведены ниже):

1. Создать окружение и установить зависимости.
2. Запустить генерацию на одном промпте или на файле с промптами и сравнить `vqa_score`, `vanilla_sd`, `flux1`.
3. Посчитать метрики по сохраненным результатам.
4. При необходимости отдельно запустить пайплайн редактирования на примерах из `datasets/coco/...`.

Автор исполнял код на `Python 3.10.20` и `CUDA 12.0/12.6/12.8` на `A100 80GB` и `RTX 6000Ada 45GB`. Для запуска `vlm guided` пайплайна генерации требуется около `39 GB VRAM`; этого объема также достаточно для всех сценариев, кроме локального подсчета метрик `quality` и `alignment`.

## Установка репозитория и создание окружения

```shell
git clone git@github.com:AlexKrachun/vlm_guidance_research.git
cd vlm_guidance_research/
conda env create -f environment.yaml
conda activate t2v
pip install flash-attn --no-build-isolation
pip install hydra-core
pip install tensorboard

```

## Запуск трех приведенных пайплайнов генерации
Для исполнения FLUX пайплайна надо авторизоваться в Hugging Face:

```shell
hf auth login
```



Запустить vanilla sd1.5, vqa guided sd1.5, flux1-dev на своем промпте
```shell
python -m vlm_guidance_project.vlm_guidance.run \
  run.prompt="a bear does handstand in the park" \
  run.vqa_score=True \
  run.vanilla_sd=True \
  run.flux1=True \
  scorer.model_name=qwen-2-5-vl-3b-instruct \
  algorithm.gd_steps=1 \
  algorithm.gd_lr=1
```


Запустить vanilla sd1.5, vqa guided sd1.5, flux1-dev на текстовом файле datasets/subset.txt - где каждая строка - это один промпт (можете вписать свои промпты)
```shell
python -m vlm_guidance_project.vlm_guidance.batch_run \
  batch.prompts_file=../datasets/subset.txt \
  batch.output_root_dir=subset_generations \
  run.vqa_score=True \
  run.vanilla_sd=True \
  run.flux1=True \
  scorer.model_name=qwen-2-5-vl-3b-instruct \
  algorithm.gd_steps=1 \
  algorithm.gd_lr=1
```


## Подсчет и визуализация метрик

### CLIP score

Посчитать и сохранить в CSV файл CLIP score по папке с изображениями, полученными разными пайплайнами (папка получена с помощью прогона `python -m vlm_guidance_project.vlm_guidance.batch_run`)
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

<!-- Для подсчета метрик alignment и quality по папке `vlm_guidance_project/your_pipelines_folder`, содержащей множество ваших batch run экспериментов
```shell
python3 metrics/alignment_local.py \
  -generations vlm_guidance_project/your_pipelines_folder \
  --output metrics/alignment_local_your_pipelines.csv \
  --summary-output metrics/alignment_local_your_pipelines_summary.csv \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --dtype bfloat16 \
  --batch-size 16 \
  --max-new-tokens 128 \
  --explanation-max-words 18 \
  --resume
```

Построить scatter plot по локальным alignment и quality:
```shell
python metrics/alignment_local_plot.py \
  --input metrics/alignment_local_your_pipelines_summary.csv \
  --output metrics/alignment_local_your_pipelines_scatter.png \
  --section-column experiment \
  --section vanilla=vanilla \
  --section qwen_base=qwen2.5-gd-1iter-lr1,qwen2.5-gd-2iter-lr1,qwen2.5-gd-10iter-lr0.2 \
  --section clip=clip-gd-1iter-lr1,clip-gd-2iter-lr1,clip-gd-10iter-lr0.2 \
  --section qwen_noise_info=qwen2.5-gd-1iter-lr1-noise-info,qwen2.5-gd-2iter-lr1-noise-info,qwen2.5-gd-10iter-lr0.2-noise-info \
  --section qwen_yes_loss=qwen2.5-gd-1iter-lr1-yes-loss,qwen2.5-gd-2iter-lr1-yes-loss,qwen2.5-gd-10iter-lr0.2-yes-loss \
  --comparison clip=vanilla,qwen_base,clip \
  --comparison noise_info=vanilla,qwen_base,qwen_noise_info \
  --comparison yes_loss=vanilla,qwen_base,qwen_yes_loss
``` -->



<!-- ### Визуализация статистик хода генерации

Сгенерировать график статистики градиентов VLM guidance
```shell
python metrics/statistics.py \
  --split-percentile 0.2 \
  --input-root vlm_guidance_project/subset_generations
``` -->


<!-- Построить график динамики VLM loss для `vqa_score` и `vanilla_sd`. Для vanilla пайплайна VLM loss считается только при `run.vanilla_calc_vlm_loss=True`:
```shell
python -m vlm_guidance_project.vlm_guidance.batch_run \
  batch.prompts_file=../datasets/subset.txt \
  batch.output_root_dir=loss_dynamics \
  run.vqa_score=True \
  run.vanilla_sd=True \
  run.vanilla_calc_vlm_loss=True
```

```shell
python metrics/vlm_loss_dynamics_plot.py \
  vlm_guidance_project/loss_dynamics \
  --output metrics/vlm_loss_dynamics.png
``` -->



## Запуск пайплайнов редактирования

Запустить null-text editing на датасете datasets/coco/first3
```shell
python -m vlm_guidance_editing.vlm_guidance.run \
  run.dataset_root=datasets/coco/first3 \
  run.pipeline_null_text_inversion=True \
  run.pipeline_vlm_guided_editing=True \
  run.output_root_dir=vlm_guidance_editing/first3_results \
  algorithm.save_debug_tensors=False \
  algorithm.gd_only_first_k_steps=15 \
  guided.gd_steps=3 \
  guided.zt_optimizing=true \
  guided.zt_lr=1 \
  guided.null_text_emb_optimizing=true \
  guided.null_text_emb_lr=1

```

<!-- 
---
### Основные Hydra флаги генерации

Основные конфиги лежат в `vlm_guidance_project/vlm_guidance/configs`. Для single prompt режима используется `configs/config.yaml`, для batch режима - `configs/batch_config.yaml`.

#### Входы и режим запуска

| Флаг | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `run.prompt` | `"a cat sitting on a chair"` | Текстовый prompt для `python -m vlm_guidance_project.vlm_guidance.run`. В `batch_config.yaml` не задан, потому что prompt берется из файла. |
| `batch.prompts_file` | `../datasets/subset.txt` | Файл с промптами для batch запуска, один prompt на строку. |
| `batch.output_root_dir` | `subset` | Корневая папка сохранения batch результатов. |
| `batch.skip_empty_lines` | `true` | Пропускать пустые строки в файле с промптами. |

#### Выбор пайплайнов

| Флаг | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `run.vqa_score` | `true` | Включает SD1.5 с VLM guidance. |
| `run.vanilla_sd` | `false` | Включает обычный SD1.5 baseline. |
| `run.flux1` | `false` | Включает `FLUX.1-dev` baseline. |
| `run.vanilla_calc_vlm_loss` | `false` | В vanilla SD1.5 считает VLM loss/score по шагам без gradient guidance. Нужен для сравнения динамики VLM loss между `vanilla_sd` и `vqa_score`. |

#### Общие параметры генерации

| Флаг | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `run.negative_prompt` | `"blurry, low quality"` | Negative prompt для SD1.5. |
| `run.height` | `512` | Высота изображения. |
| `run.width` | `512` | Ширина изображения. |
| `run.num_inference_steps` | `30` | Число denoising шагов. |
| `run.guidance_scale` | `7.5` | Classifier-free guidance scale внутри SD1.5. |
| `run.seed` | `42` | Seed генерации. |
| `run.batch_size` | `1` | Размер batch внутри одного prompt. |

#### VLM scorer и loss

| Флаг | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `scorer.model_name` | `clip-flant5-xxl` | Scorer для VLM guidance. Для Qwen2.5-VL используйте `qwen-2-5-vl-3b-instruct`. |
| `scorer.device` | `cuda:0` | Устройство для scorer. |
| `run.verbose_vlm` | `false` | Дополнительно просит VLM сгенерировать текстовое объяснение по промежуточному изображению и сохраняет его в `summary.json`/TensorBoard. |
| `run.vlm_num_tokens` | `50` | Максимум новых токенов для `verbose_vlm`. |
| `run.vqa_vlm_prompt_template` | `'Does this figure show "{}"? Please answer yes or no'` | Шаблон вопроса для VLM score/loss; должен содержать `{}` для подстановки prompt. |
| `run.verbose_vlm_prompt_template` | `'Describe whether this image matches the prompt  "{}". Explain briefly what matches and what does not.'` | Шаблон вопроса для verbose ответа. |
| `run.yes_no_loss` | `true` | Выбирает loss для VLM guidance. Формулы приведены ниже. |

`run.yes_no_loss` по умолчанию равен `true` и выбирает, какой VLM loss используется для guidance:

- `run.yes_no_loss=true`: `Yes/No` margin loss  
  $\mathcal{L}_{\text{vlm}} = -\log(\sigma(p(\text{"Yes"}) - p(\text{"No"})))$
- `run.yes_no_loss=false`: cross-entropy loss на токен `Yes`  
  $\mathcal{L}_{\text{vlm}} = \text{CrossEntropy}(\text{VLM}(I, q(p)), \text{"Yes"})$

#### Gradient guidance

| Флаг | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `algorithm.gd_steps` | `2` | Число gradient descent шагов на одном denoising step. |
| `algorithm.gd_lr` | `0.5` | Learning rate для обновления latent. |
| `algorithm.gd_only_first_k_steps` | `10` | Guidance применяется только на первых `k` denoising шагах. |
| `algorithm.normalize_grad` | `true` | Нормировать градиент перед обновлением latent. |
| `algorithm.clamp_grad_value` | `1.0` | Ограничить значения градиента перед обновлением latent. |
| `algorithm.save_debug_tensors` | `false` | Сохраняет промежуточные `xt`, `x0` и diff-картинки для отладки. |

---
По вопросам обращайтесь ко мне в телеграм: [@Alex_Karachun](http://t.me/Alex_Karachun) -->
