# Лабораторна робота №1
## "Безпека даних в ML"
## Ходаков Максим Олегович ШІ-2

## Постановка задачі

1. Обрано архітектуру **ResNet-18** (`torchvision.models.resnet18`, попередньо натреновану на ImageNet-1k)
   і дотреновано (fine-tuning) на підвибірці **imagenet10** (10 класів, по 1300 зображень кожен).
2. Модель натреновано так, щоб вона була стійкою одночасно до:
   - **FGSM-атаки** (Fast Gradient Sign Method)
   - **Gaussian Noise атаки** (μ=0, σ ~ U(0.01, 0.08), p=0.5)
3. Для FGSM знайдено межове значення `eps`, при якому точність моделі падає до рівня випадкового
   вгадування (руйнування моделі).

## Структура проєкту

```
src/
  data.py       — завантаження датасету, стратифікований train/val split (85/15)
  model.py      — ResNet-18 + шар нормалізації (вхід моделі — [0,1] пікселі)
  attacks.py    — реалізація FGSM та Gaussian Noise атак
  train.py      — тренування (baseline / adversarially-robust)
  evaluate.py   — оцінка: clean accuracy, noise accuracy, FGSM epsilon sweep
  runs/         — чекпоінти та результати (в .gitignore)
imagenet-10/    — датасет (в .gitignore)
```

## Методика тренування (robust-модель)

На кожному кроці тренування комбінуються три гілки втрат:

- **clean** — оригінальні зображення
- **noisy** — зображення з адитивним гаусовим шумом (μ=0, σ ~ U(0.01, 0.08), застосовується з p=0.5)
- **adversarial** — FGSM-приклади (один крок, `eps` випадково обирається з `U(2/255, 8/255)` на
  кожному кроці тренування, атака будується відносно поточної (noisy) моделі)

```
loss = w_clean * CE(model(x), y)
     + w_noise * CE(model(x_noisy), y)
     + w_adv   * CE(model(x_adv), y)
```

Ваги (`w_clean = w_noise = w_adv ≈ 0.33`) рівномірні за замовчуванням.

FGSM та шумова атака реалізовані у пікселях `[0,1]` (нормалізація ImageNet-mean/std винесена
всередину моделі, першим шаром), тому значення `eps` мають звичну інтерпретацію (наприклад
`8/255 ≈ 0.031`).

## Як відтворити

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Тренування robust-моделі
python src/train.py --data-root imagenet-10 --out-dir src/runs/robust --epochs 6

# Тренування baseline-моделі (без adversarial/noise тренування) для порівняння
python src/train.py --data-root imagenet-10 --out-dir src/runs/baseline \
    --epochs 6 --w-clean 1.0 --w-noise 0 --w-adv 0

# Оцінка (clean accuracy, noise accuracy, FGSM epsilon sweep, пошук межі eps)
python src/evaluate.py --checkpoint src/runs/robust/best_model.pt --out-dir src/runs/robust
python src/evaluate.py --checkpoint src/runs/baseline/best_model.pt --out-dir src/runs/baseline
```
