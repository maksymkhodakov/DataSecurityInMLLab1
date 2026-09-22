"""
ТОЧКА ВХОДУ для ОЦІНКИ вже натренованої моделі (запускати напряму: `python evaluate.py ...`).

Що робить цей скрипт:
  1. Завантажує checkpoint моделі (збережений train.py) і валідаційну частину датасету.
  2. Рахує "чисту" точність (без жодних атак) — базовий рівень якості моделі.
  3. Рахує точність під Gaussian Noise атакою (шумова атака, така сама, що й на тренуванні).
  4. Робить EPSILON SWEEP для FGSM: прогонятиме атаку з різними значеннями eps
     (від 0 до дуже великого) і для кожного eps рахує точність на валідації.
     Саме тут знаходиться відповідь на вимогу лабораторної:
     "знайти ліміт eps при якому модель перестає працювати".
  5. Зберігає результати у JSON (eval_results.json) та графік точність-vs-eps (fgsm_sweep.png).

Запуск:
    python evaluate.py --checkpoint runs/robust/best_model.pt --out-dir runs/robust
"""
import argparse
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # без GUI-бекенду, щоб працювало у фоновому/термінальному режимі
import matplotlib.pyplot as plt

from data import get_dataloaders
from model import build_model
from attacks import fgsm_attack, gaussian_noise_attack
from train import get_device


def load_model(checkpoint_path, device):
    """Відновлює модель із checkpoint-файлу (.pt), який зберігає train.py.

    checkpoint містить:
      - "model_state": ваги моделі (state_dict)
      - "classes": список назв класів (WordNet ID папок imagenet-10),
        порядок = порядок вихідних нейронів моделі (важливо для argmax -> label).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    # pretrained=False: ваги backbone все одно перезапишуться з checkpoint,
    # тому завантажувати їх з інтернету тут не потрібно (швидше і працює офлайн).
    model = build_model(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # вимикає dropout/batchnorm-статистику -> детерміновані передбачення
    return model, classes


def accuracy_clean(model, loader, device):
    """Точність моделі на оригінальних (без атаки) зображеннях."""
    correct, total = 0, 0
    with torch.no_grad():  # градієнти тут не потрібні -> економимо пам'ять/час
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def accuracy_fgsm(model, loader, device, eps):
    """Точність моделі, коли КОЖНЕ зображення батчу спочатку атакується FGSM
    з переданим eps, а потім подається на класифікацію.

    eps — це максимальна L_inf-величина збурення пікселя, в одиницях [0,1]
    (тобто eps=1/255 означає зміну яскравості пікселя щонайбільше на 1 з 255
    можливих рівнів; eps=8/255 ≈ 0.031 — типове "помірне" значення в літературі
    з adversarial robustness; eps=1.0 означає, що піксель можна зіпсувати
    повністю, від 0 до 1).

    Важливо: fgsm_attack() сам рахує градієнт loss відносно входу x
    (потребує requires_grad), тому виклик НЕ обгорнутий у torch.no_grad() —
    но_grad вмикається лише під час самого forward-pass для передбачення
    вже атакованого x_adv.
    """
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = fgsm_attack(model, x, y, eps=eps)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def accuracy_noise(model, loader, device, sigma_low, sigma_high, p, trials=3):
    """Точність моделі під Gaussian Noise атакою.

    sigma для кожного зображення обирається випадково з U(sigma_low, sigma_high),
    а сам шум додається лише з ймовірністю p (це узгоджено з тим, як атака
    застосовувалась під час тренування: mu=0, sigma~U(0.01,0.08), p=0.5).

    Тут p=1.0 передається з main(), тобто атака застосовується до КОЖНОГО
    зображення валідаційного набору (найгірший, а не середній випадок).

    Оскільки шум — випадковий, точність вимірюється `trials` разів (за
    замовчуванням 3) з різними випадковими шумами і усереднюється, щоб
    результат не залежав від одного випадкового "щасливого"/"невдалого" прогону.
    """
    accs = []
    for _ in range(trials):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                x_noisy = gaussian_noise_attack(x, sigma_low, sigma_high, p)
                preds = model(x_noisy).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accs.append(correct / total)
    return sum(accs) / len(accs)


def find_break_eps(eps_values, accs, num_classes, random_thresh_factor=1.5, half_thresh=0.5):
    """Визначає межові значення eps ("точки зламу" моделі) з кривої eps -> accuracy.

    Оскільки "модель перестала працювати" — поняття без єдиного формального
    визначення, тут рахуються ДВІ різні межі (обидві виводяться в звіт, щоб
    показати повну картину, а не одну довільну цифру):

    1) eps_break_half — перше (найменше) значення eps, при якому точність
       падає нижче 50% від чистої (clean) точності моделі. Це "практична"
       межа: модель ще працює, але вже вдвічі гірше за нормальний режим.

    2) eps_break_random — перше значення eps, при якому точність опускається
       майже до рівня випадкового вгадування (1/num_classes, для 10 класів
       це 10%). Тут random_thresh_factor=1.5 дає невеликий запас (тобто
       поріг = 15% для 10 класів), бо точність рідко падає рівно до 10.00%
       через шум оцінки на скінченній вибірці. Це "повна" межа — модель
       фактично вгадує наосліп.

    Якщо жодне зі значень eps зі списку не дає такого падіння, відповідне
    значення залишається None (означає: потрібно розширити eps_list —
    модель не зламалась навіть на максимальному перевіреному eps).
    """
    random_baseline = 1.0 / num_classes
    clean_acc = accs[0]  # припущення: eps_values[0] == 0.0 (перше значення в списку)

    eps_break_random = None
    eps_break_half = None
    for e, a in zip(eps_values, accs):
        if eps_break_half is None and a < half_thresh * clean_acc:
            eps_break_half = e
        if eps_break_random is None and a <= random_baseline * random_thresh_factor:
            eps_break_random = e
    return eps_break_half, eps_break_random


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    # val_loader береться з того самого стратифікованого split (seed узгоджений
    # з train.py), тому це саме ті зображення, які модель НЕ бачила під час тренування.
    _, val_loader, classes = get_dataloaders(
        args.data_root, batch_size=args.batch_size, val_fraction=args.val_fraction,
        seed=args.seed, num_workers=args.workers,
    )
    model, classes = load_model(args.checkpoint, device)
    num_classes = len(classes)

    results = {}

    # --- 1) Базова (чиста) точність ---
    clean_acc = accuracy_clean(model, val_loader, device)
    results["clean_acc"] = clean_acc
    print(f"Clean accuracy: {clean_acc:.4f}")

    # --- 2) Стійкість до шумової атаки (Gaussian Noise) ---
    noise_acc = accuracy_noise(model, val_loader, device, 0.01, 0.08, p=1.0, trials=args.noise_trials)
    results["noise_acc_p1"] = noise_acc
    print(f"Gaussian noise accuracy (p=1.0, sigma~U(0.01,0.08), avg of {args.noise_trials} trials): {noise_acc:.4f}")

    # --- 3) EPSILON SWEEP для FGSM ---
    # eps_list (див. parse_args нижче) охоплює діапазон від 0 (без атаки) до
    # дуже великих значень (128/255 ≈ 0.5 — половина всього діапазону пікселя),
    # щоб гарантовано "накрити" точку, де модель ламається, навіть якщо вона
    # дуже стійка. Для кожного eps окремо прогоняється весь val_loader.
    eps_values = [float(e) for e in args.eps_list]
    accs = []
    for eps in eps_values:
        a = accuracy_fgsm(model, val_loader, device, eps)
        accs.append(a)
        # eps друкується одразу в двох форматах: як частка [0,1] і як N/255
        # (стандартна одиниця вимірювання в adversarial ML літературі)
        print(f"FGSM eps={eps:.4f} ({eps*255:.1f}/255) -> accuracy={a:.4f}")

    results["fgsm_eps_values"] = eps_values
    results["fgsm_accs"] = accs

    # --- 4) Пошук межового eps ("ліміт, при якому модель перестає працювати") ---
    eps_break_half, eps_break_random = find_break_eps(eps_values, accs, num_classes)
    results["eps_break_below_50pct_of_clean"] = eps_break_half
    results["eps_break_near_random_guessing"] = eps_break_random

    print(f"\nEps where accuracy drops below 50% of clean accuracy: {eps_break_half}")
    print(f"Eps where accuracy collapses to ~random guessing ({1/num_classes:.3f}): {eps_break_random}")

    # --- 5) Збереження результатів ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Графік: вісь X — eps у одиницях /255 (звичніше читати, ніж дроби [0,1]),
    # вісь Y — точність. Горизонтальна пунктирна лінія — рівень випадкового
    # вгадування (1/num_classes), орієнтир для "повного зламу" моделі.
    plt.figure(figsize=(7, 5))
    plt.plot([e * 255 for e in eps_values], accs, marker="o")
    plt.axhline(1.0 / num_classes, color="gray", linestyle="--", label="random guessing")
    plt.xlabel("FGSM epsilon (in /255 units)")
    plt.ylabel("Accuracy")
    plt.title("FGSM robustness sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "fgsm_sweep.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_dir / 'fgsm_sweep.png'}")
    print(f"Saved results to {out_dir / 'eval_results.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="imagenet-10")
    p.add_argument("--checkpoint", type=str, default="src/runs/exp1/best_model.pt")
    p.add_argument("--out-dir", type=str, default="src/runs/exp1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--noise-trials", type=int, default=3)
    # Список значень eps для sweep. Нелінійна сітка: густіша в "цікавому"
    # низькому діапазоні (0..16/255, де зазвичай і відбувається злам
    # неробастних моделей), і рідша у високому (аж до 128/255) — щоб
    # гарантовано побачити повний колапс точності навіть у дуже стійкої моделі.
    p.add_argument("--eps-list", type=float, nargs="+", default=[
        0.0, 1/255, 2/255, 4/255, 6/255, 8/255, 12/255, 16/255,
        24/255, 32/255, 48/255, 64/255, 96/255, 128/255,
    ])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
