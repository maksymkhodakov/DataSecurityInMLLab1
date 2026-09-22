"""
ТОЧКА ВХОДУ для ТРЕНУВАННЯ моделі (запускати напряму: `python train.py ...`).

Ідея: замість стандартного тренування "clean images -> loss -> backprop"
на кожному кроці рахуються ТРИ окремі forward-passes і три компоненти
loss, які потім складаються в одну зважену суму:

  1. clean   — оригінальні зображення (базова якість класифікації)
  2. noisy   — ті самі зображення + Gaussian Noise атака
               (mu=0, sigma ~ U(0.01, 0.08), застосовується з ймовірністю p=0.5)
  3. adv     — FGSM-adversarial приклади, побудовані з noisy-версії зображень
               з випадковим eps на кожному кроці (U(eps_min, eps_max))

  loss = w_clean * CE(model(x),       y)
       + w_noise * CE(model(x_noisy), y)
       + w_adv   * CE(model(x_adv),   y)

Це "adversarial training" (Goodfellow et al., 2015 / Madry et al., 2018,
спрощена одно-крокова FGSM-версія) поєднана з data augmentation шумом —
модель бачить під час тренування ті самі типи спотворень, з якими її
будуть атакувати під час оцінки (evaluate.py), і вчиться давати правильну
відповідь навіть на спотворених входах.

Якщо потрібна ЗВИЧАЙНА (не robust) модель для порівняння (baseline),
достатньо запустити з --w-clean 1.0 --w-noise 0 --w-adv 0 — тоді гілки
noisy/adv взагалі не рахуються (для швидкості).
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders
from model import build_model
from attacks import fgsm_attack, gaussian_noise_attack


def get_device():
    """Апаратне прискорення: MPS (Apple Silicon GPU) > CUDA > CPU (fallback)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_clean(model, loader, device):
    """Точність на оригінальних (без атак) зображеннях — контроль базової якості."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_fgsm_quick(model, loader, device, eps):
    """Швидка перевірка FGSM-стійкості в кінці кожної епохи (одне фіксоване
    значення eps = args.eps_max, а не повний sweep — повний sweep з багатьма
    eps робить evaluate.py окремо, після тренування)."""
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = fgsm_attack(model, x, y, eps=eps)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # Стратифікований train/val split (85/15 за замовчуванням) із фіксованим
    # seed — гарантує, що evaluate.py пізніше побачить ТОЙ САМИЙ val-набір
    # (жодне валідаційне зображення не потрапляє в тренування).
    train_loader, val_loader, classes = get_dataloaders(
        args.data_root, batch_size=args.batch_size, val_fraction=args.val_fraction,
        seed=args.seed, num_workers=args.workers,
    )
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ResNet-18, попередньо натренована на ImageNet-1k (pretrained=True),
    # з заміненим останнім шаром на 10 виходів (кількість класів imagenet-10).
    model = build_model(num_classes=len(classes), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # --- Гілка 1: clean loss (завжди рахується) ---
            loss = args.w_clean * F.cross_entropy(model(x), y)

            # --- Гілка 2: noise-augmented loss ---
            # Gaussian Noise застосовується per-sample з ймовірністю p=0.5
            # (частина зображень у батчі залишається чистою, частина — зашумленою),
            # точно за умовою лабораторної: mu=0, sigma ~ U(0.01, 0.08), p=0.5.
            if args.w_noise > 0:
                x_noisy = gaussian_noise_attack(x, sigma_low=0.01, sigma_high=0.08, p=0.5)
                loss = loss + args.w_noise * F.cross_entropy(model(x_noisy), y)
            else:
                x_noisy = x  # без noise-гілки FGSM (нижче) атакує "чисті" x

            # --- Гілка 3: adversarial (FGSM) loss ---
            if args.w_adv > 0:
                # FGSM атакує вже NOISY-версію входу (x_noisy), а не тільки
                # чистий x, — модель бачить "найгірший" з двох типів спотворень
                # одразу і вчиться бути стійкою до їхньої комбінації.
                #
                # eps на тренуванні НЕ фіксований, а випадковий на кожному
                # кроці: U(eps_min, eps_max) = U(2/255, 8/255) за замовчуванням.
                # Це стандартний прийом (random eps) — робить модель стійкою
                # до ДІАПАЗОНУ силу атаки, а не тільки до одного конкретного eps.
                model.eval()  # атака будується без dropout/BN-шуму в forward
                train_eps = torch.empty(1).uniform_(args.eps_min, args.eps_max).item()
                x_adv = fgsm_attack(model, x_noisy, y, eps=train_eps)
                model.train()
                loss = loss + args.w_adv * F.cross_entropy(model(x_adv), y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                print(f"epoch {epoch} step {step+1}/{len(train_loader)} "
                      f"loss {running_loss / (step+1):.4f}")

            if args.max_steps and (step + 1) >= args.max_steps:
                break  # лише для дебагу/швидкого smoke-тесту, за замовчуванням вимкнено (0)

        scheduler.step()

        # --- Валідація в кінці епохи ---
        clean_acc = evaluate_clean(model, val_loader, device)
        # Швидка перевірка стійкості до FGSM з eps=eps_max (найсильніша атака
        # з тренувального діапазону) — не повний sweep, лише індикатор прогресу
        # між епохами. Повний sweep по багатьох eps робить окремо evaluate.py.
        adv_acc = evaluate_fgsm_quick(model, val_loader, device, eps=args.eps_max)

        dt = time.time() - t0
        print(f"[epoch {epoch}] loss={running_loss/len(train_loader):.4f} "
              f"clean_acc={clean_acc:.4f} fgsm_acc(eps={args.eps_max:.3f})={adv_acc:.4f} "
              f"time={dt:.1f}s")

        # Модель, що потрапляє в best_model.pt, обирається за комбінованим
        # критерієм (60/50 балансу clean/adv), а не лише за clean-точністю —
        # інакше найкращою могла б вважатись epoch, де модель "забула" про
        # робастність заради +1% чистої точності.
        score = 0.5 * clean_acc + 0.5 * adv_acc
        if score > best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "clean_acc": clean_acc,
                "fgsm_acc": adv_acc,
            }, out_dir / "best_model.pt")
            print(f"  -> saved new best model (score={score:.4f})")

    # Модель з останньої епохи зберігається окремо (для порівняння/дебагу),
    # основним артефактом для evaluate.py є саме best_model.pt.
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
    }, out_dir / "last_model.pt")
    print("Training complete.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="imagenet-10")
    p.add_argument("--out-dir", type=str, default="src/runs/exp1")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    # Діапазон eps, з якого випадково обирається сила FGSM-атаки на кожному
    # тренувальному кроці (не єдине фіксоване значення — див. коментар вище).
    p.add_argument("--eps-min", type=float, default=2 / 255)
    p.add_argument("--eps-max", type=float, default=8 / 255)
    # Ваги трьох компонент loss. За замовчуванням майже рівні (~1/3 кожна),
    # щоб модель однаково "цінувала" чисту точність, стійкість до шуму й
    # стійкість до FGSM. Для baseline-моделі (без робастності) використовуйте
    # --w-clean 1.0 --w-noise 0 --w-adv 0.
    p.add_argument("--w-clean", type=float, default=0.34)
    p.add_argument("--w-noise", type=float, default=0.33)
    p.add_argument("--w-adv", type=float, default=0.33)
    p.add_argument("--max-steps", type=int, default=0, help="debug: cap steps per epoch")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
