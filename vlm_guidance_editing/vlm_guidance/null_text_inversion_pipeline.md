# Null-Text Inversion Pipeline for Image Editing

Этот файл фиксирует полный пайплайн `null-text inversion` в проекте `prompt-to-prompt`, а также то, как он стыкуется с последующим `prompt-to-prompt` редактированием реального изображения.

Основные первоисточники:
- `prompt-to-prompt/README.md`
- `prompt-to-prompt/null_text_w_ptp.ipynb`
- `prompt-to-prompt/ptp_utils.py`

## 1. Что вообще делает null-text inversion

Задача: взять реальное изображение `x`, найти для него латентный шумовой якорь `x_T` и такую последовательность unconditional/null-text embeddings
`{e_null^t}_{t=1..T}`,
чтобы Stable Diffusion могла:
- хорошо реконструировать исходное изображение по исходному тексту;
- затем редактировать то же изображение, если заменить только conditional prompt и/или вмешаться в attention через Prompt-to-Prompt.

Ключевая идея статьи и notebook:
- сначала делается **DDIM inversion** изображения в латентное шумовое состояние;
- затем **замораживается модель** и **оптимизируется только unconditional embedding** (`""`, null-text embedding) отдельно для каждого шага денойзинга;
- после этого при генерации используется:
  - тот же стартовый латент `x_T`,
  - та же последовательность оптимизированных null-text embeddings,
  - новый conditional prompt,
  - и контроллер attention из Prompt-to-Prompt.

## 2. Полный пайплайн целиком

### Шаг 0. Инициализация Stable Diffusion

В notebook создаются:
- `DDIMScheduler(...)`
- `StableDiffusionPipeline.from_pretrained(...)`
- глобальные параметры:
  - `NUM_DDIM_STEPS = 50`
  - `GUIDANCE_SCALE = 7.5`
  - `MAX_NUM_WORDS = 77`

Это означает, что весь inversion/editing пайплайн работает в DDIM-режиме на 50 шагах.

### Шаг 1. Подготовка входного изображения

Функция `load_512(...)`:
- читает изображение;
- применяет опциональные crop offsets;
- центрирует в квадрат;
- ресайзит до `512 x 512`.

На выходе получается numpy-изображение, совместимое со Stable Diffusion v1.x.

### Шаг 2. Кодирование изображения в VAE latent

Метод `image2latent(...)`:
- переводит изображение в диапазон `[-1, 1]`;
- прогоняет через `VAE.encode(image).latent_dist.mean`;
- масштабирует latent на коэффициент `0.18215`.

Обозначим этот латент как `z_0`.

Это не шум, а VAE-представление исходного изображения.

### Шаг 3. Кодирование текста

Метод `init_prompt(prompt)` строит два embedding-а:
- `e_null`: embedding пустой строки `""`;
- `e_cond`: embedding исходного текстового описания изображения.

Дальше они конкатенируются в `context = [e_null; e_cond]`.

Важно: в оригинальном null-text inversion **conditional embedding фиксирован**, а оптимизируется только null/unconditional embedding.

### Шаг 4. DDIM inversion: переход от `z_0` к шумовому якорю `z_T`

Метод `ddim_inversion(image)` делает две вещи:
- сохраняет `image_rec = latent2image(z_0)` как простую VAE-реконструкцию;
- запускает `ddim_loop(z_0)`.

#### Что делает `ddim_loop`

`ddim_loop` строит цепочку латентов:
- `z_0, z_1, z_2, ..., z_T`

На каждом шаге:
- берётся timestep в обратном порядке относительно обычной генерации;
- UNet предсказывает шум **только по conditional embedding** `e_cond`;
- затем вызывается `next_step(...)`, которая аналитически переводит латент на следующий, более шумный шаг DDIM.

То есть inversion идёт так:
- старт: чистый VAE latent `z_0`;
- конец: шумовой latent `z_T`, который потом станет стартовой точкой редактирования.

Важно:
- в notebook DDIM inversion использует только `e_cond`, без оптимизированного null-text;
- это даёт грубый якорь, но обычно reconstruction ещё не идеальна.

### Шаг 5. Null-text optimization

Это главный шаг метода.

Метод `null_optimization(latents, num_inner_steps, epsilon)` получает:
- всю DDIM-цепочку `latents = [z_0, z_1, ..., z_T]`;
- число внутренних шагов оптимизации на timestep;
- критерий ранней остановки.

#### Идея

Для каждого шага денойзинга `t_i` мы хотим подобрать свой unconditional embedding `e_null^i`, чтобы один CFG-шаг из текущего латента восстанавливал следующий target из DDIM inversion максимально точно.

Иначе говоря, для каждого timestep решается локальная задача:
- известен текущий latent `z_i`;
- известен target предыдущего шага `z_{i-1}` из DDIM inversion;
- фиксирован `e_cond`;
- оптимизируется только `e_null^i`.

#### Что происходит внутри одного timestep

Для шага `i`:
- берётся текущий латент `latent_cur`;
- берётся target `latent_prev` из DDIM inversion;
- вычисляется `noise_pred_cond = UNet(latent_cur, t_i, e_cond)`;
- создаётся оптимизируемая копия `uncond_embeddings`.

Дальше внутренний цикл:
1. `noise_pred_uncond = UNet(latent_cur, t_i, e_null^i)`
2. classifier-free guidance:
   `noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)`
3. выполняется обратный DDIM-переход `prev_step(noise_pred, t_i, latent_cur)`
4. получается реконструкция предыдущего латента `latent_prev_rec`
5. считается loss:
   `MSE(latent_prev_rec, latent_prev)`
6. градиент идёт **только в `e_null^i`**
7. если loss достаточно мал, включается early stop

После оптимизации:
- сохраняется найденный `e_null^i`;
- затем текущий latent обновляется уже с этим оптимизированным unconditional embedding, чтобы перейти к следующему timestep.

#### Что в итоге сохраняется

На выходе получаем последовательность:
- `E_null = [e_null^1, e_null^2, ..., e_null^T]`

Это и есть ядро null-text inversion.

## 3. Что возвращает inversion

Метод `invert(...)` в notebook возвращает:
- `(image_gt, image_rec)`
- `x_T = ddim_latents[-1]`
- `uncond_embeddings` — список оптимизированных null-text embeddings по timesteps

То есть после inversion у нас есть всё необходимое для дальнейшего редактирования.

## 4. Как выполняется реконструкция/генерация после inversion

В notebook переопределяется `text2image_ldm_stable(...)`, чтобы на каждом шаге денойзинга можно было подать **свой** unconditional embedding.

Обычная Stable Diffusion использует один и тот же `e_null` на всех шагах.
Здесь используется последовательность `E_null[i]`.

Цикл генерации делает следующее:
1. стартует из `x_T`;
2. токенизирует prompt(ы) и получает `text_embeddings`;
3. на каждом timestep `t_i` собирает context:
   - либо стандартный `e_null`,
   - либо оптимизированный `E_null[i]`;
4. вызывает `ptp_utils.diffusion_step(...)`;
5. в конце декодирует латенты в изображение.

Если prompt остаётся исходным, это даёт реконструкцию.
Если prompt меняется, это уже редактирование.

## 5. Где здесь Prompt-to-Prompt

Null-text inversion сам по себе лишь делает реальное изображение редактируемым внутри Stable Diffusion.
Собственно редактирование делает Prompt-to-Prompt через контроль attention.

### Регистрация attention hook-ов

`ptp_utils.register_attention_control(model, controller)`:
- обходит cross-attention слои UNet;
- подменяет их `forward`;
- в каждом attention слое передаёт attention map в controller.

### Базовые контроллеры из notebook

Используются классы:
- `AttentionStore`
- `AttentionControlEdit`
- `AttentionReplace`
- `AttentionRefine`
- `AttentionReweight`
- `LocalBlend`

Их роль:
- `AttentionReplace`: заменить cross-attention старого prompt на attention нового prompt для token replacement;
- `AttentionRefine`: мягко уточнять attention при добавлении новых слов;
- `AttentionReweight`: усиливать/ослаблять вклад отдельных слов;
- `LocalBlend`: ограничивать редактирование локальной областью изображения по attention map.

### Как это сочетается с inversion

Финальный editing pipeline выглядит так:
1. взять реальное изображение;
2. сделать `invert(image, source_prompt)`;
3. получить `x_T` и `E_null`;
4. задать `edit_prompt`;
5. построить controller через `make_controller(...)`;
6. запустить denoising из `x_T`:
   - conditional embedding уже от `edit_prompt`;
   - unconditional embedding на шаге `i` равен `E_null[i]`;
   - внутри UNet attention модифицируется controller-ом.

Именно это превращает Prompt-to-Prompt из метода редактирования синтетических генераций в метод редактирования **реальных изображений**.

## 6. Весь пайплайн в виде короткой схемы

```text
real image
  -> crop/resize to 512x512
  -> VAE encode
  -> z_0
  -> DDIM inversion with source prompt
  -> [z_0, z_1, ..., z_T]
  -> optimize null-text embedding separately for each timestep
  -> E_null = [e_null^1, ..., e_null^T]
  -> start from z_T
  -> choose edit prompt
  -> run reverse denoising with:
       - conditional embedding from edit prompt
       - unconditional embedding E_null[i] at each step
       - optional Prompt-to-Prompt attention control
  -> edited image
```

## 7. Псевдокод полного процесса

```python
# inversion
image = load_512(image_path)
z0 = VAE.encode(image)
e_null = TextEncoder("")
e_cond_src = TextEncoder(source_prompt)

# DDIM inversion anchor
z = z0
ddim_latents = [z0]
for t in reversed(train_timesteps_for_sampling):
    eps_cond = UNet(z, t, e_cond_src)
    z = next_step(eps_cond, t, z)
    ddim_latents.append(z)
zT = ddim_latents[-1]

# null-text optimization
optimized_nulls = []
latent_cur = zT
for i, t in enumerate(forward_sampling_timesteps):
    target_prev = ddim_latents[-i - 2]
    e_null_i = clone(e_null)
    optimize e_null_i to minimize:
        MSE(
            prev_step(
                eps_uncond_cfg(latent_cur, t, e_null_i, e_cond_src),
                t,
                latent_cur,
            ),
            target_prev,
        )
    optimized_nulls.append(e_null_i)
    latent_cur = prev_step(
        eps_uncond_cfg(latent_cur, t, e_null_i, e_cond_src),
        t,
        latent_cur,
    )

# editing
z = zT
e_cond_edit = TextEncoder(edit_prompt)
for i, t in enumerate(forward_sampling_timesteps):
    context = [optimized_nulls[i], e_cond_edit]
    eps = UNet_with_attention_control(z, t, context)
    z = prev_step(eps, t, z)

edited = VAE.decode(z)
```

## 8. Что соответствует этому пайплайну в вашем проекте

По артефактам в `vlm_guidance_research_dev/outputs/null_text_inversion/...` видно, что у вас уже сохраняются типичные компоненты этого пайплайна:
- `input_resized.png` — подготовленное входное изображение после resize/crop;
- `image_latent.pt` — VAE latent исходного изображения (`z_0`);
- `ddim_latents.pt` — DDIM trajectory, обычно длины `51` при `50` шагах (`z_0 ... z_T`);
- `null_text_embeddings.pt` — последовательность оптимизированных null-text embeddings длины `50`;
- `reconstruction.png` — reconstruction после inversion;
- `reconstruction_abs_diff.png` — разница между входом и reconstruction;
- `metadata.json` — метаданные, включая `source_prompt`, число латентов и число null-text embeddings.

То есть на уровне артефактов ваш проект следует той же логике, что и оригинальный notebook:
- сначала inversion;
- потом сохранение якоря и оптимизированных unconditional embeddings;
- затем эти данные можно использовать для последующего editing.

## 9. Самая важная мысль

Null-text inversion не меняет веса Stable Diffusion и не оптимизирует conditional prompt embedding.
Она делает только одно:
- для каждого шага денойзинга подбирает такой unconditional embedding, чтобы trajectory реального изображения стала совместимой с CFG-денойзингом.

После этого Prompt-to-Prompt уже может надёжно редактировать реальное изображение почти так же, как если бы оно изначально было сгенерировано самой моделью.

## 10. Практическая формула пайплайна в одну строку

```text
real image -> VAE latent -> DDIM inversion anchor -> per-timestep null-text optimization -> reverse denoising from x_T with edited prompt + Prompt-to-Prompt attention control -> edited real image
```
