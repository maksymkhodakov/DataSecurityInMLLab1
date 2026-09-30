"""
Завантаження та розбиття датасету imagenet-10.

Зображення навмисно НЕ нормалізуються тут (лишаються у [0, 1] після
ToTensor()) — нормалізація перенесена в model.py (перший шар моделі), щоб
FGSM/Gaussian Noise атаки (attacks.py) працювали у природному діапазоні
пікселів, де eps і sigma мають прямий фізичний сенс.
"""
import random
from collections import defaultdict

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

IMG_SIZE = 224  # стандартний вхідний розмір для ResNet, натренованого на ImageNet


def build_transforms():
    """Трансформи датасету. Обидва закінчуються ToTensor() і БЕЗ Normalize
    (нормалізація — усередині моделі, див. model.py)."""
    train_tf = transforms.Compose([
        # RandomResizedCrop + horizontal flip — стандартна аугментація для
        # тренування, зменшує перенавчання на маленькому датасеті (11k зображень).
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # PIL [0,255] -> torch.FloatTensor [0,1]
    ])
    eval_tf = transforms.Compose([
        # Для валідації/тесту аугментація не потрібна — детерміноване
        # resize+center-crop, щоб оцінка була відтворюваною.
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])
    return train_tf, eval_tf


def stratified_split(dataset, val_fraction=0.15, seed=42):
    """Розбиває ImageFolder-датасет на train/val індекси, СТРАТИФІКОВАНО
    (тобто val_fraction застосовується ОКРЕМО в межах кожного класу, а не до
    всього датасету одразу) — гарантує, що всі 10 класів представлені у
    val-наборі приблизно порівну, навіть якщо в оригінальних папках трохи
    різна кількість зображень.

    seed фіксує розбиття: той самий seed у train.py і evaluate.py означає,
    що val-набір у обох скриптах — це РІВНО ті самі зображення (модель
    ніколи не тренувалась і не валідувалась на тестових зображеннях).
    """
    # Крок 1: групуємо індекси зображень за класом.
    # dataset.samples — список пар (шлях_до_файлу, індекс_класу); самі
    # зображення при цьому не читаються з диска, тож це дуже швидко.
    by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    # Окремий локальний генератор випадкових чисел (а не глобальний random),
    # щоб розбиття залежало ЛИШЕ від seed і не "збивалось" від інших викликів
    # random деінде в програмі.
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label, indices in by_class.items():
        indices = indices[:]  # копія, щоб не перемішувати оригінальний список
        rng.shuffle(indices)
        # Кількість val-зображень для ЦЬОГО класу; max(1, ...) гарантує, що
        # навіть дуже малий клас матиме хоча б одне зображення у валідації.
        n_val = max(1, int(len(indices) * val_fraction))
        val_idx.extend(indices[:n_val])     # перші n_val -> валідація
        train_idx.extend(indices[n_val:])   # решта -> тренування
    # Множини train_idx і val_idx не перетинаються за побудовою.
    return train_idx, val_idx


def get_datasets(root, val_fraction=0.15, seed=42):
    """Повертає (train_ds, val_ds, classes).

    root — папка у форматі ImageFolder: root/<клас>/<зображення>.jpg,
    де назва кожної підпапки (WordNet-ID, напр. n01440764) стає назвою класу.
    """
    train_tf, eval_tf = build_transforms()

    # base_for_split (без transform) використовується лише для читання
    # списку (шлях, клас) і побудови стратифікованого розбиття по індексах —
    # самі зображення тут не завантажуються.
    base_for_split = datasets.ImageFolder(root)
    train_idx, val_idx = stratified_split(base_for_split, val_fraction, seed)

    # Два окремих ImageFolder з РІЗНИМИ transform (train має аугментацію,
    # val — ні), обидва вказують на ту саму папку; Subset потім вибирає
    # з кожного лише "свою" половину індексів.
    train_full = datasets.ImageFolder(root, transform=train_tf)
    val_full = datasets.ImageFolder(root, transform=eval_tf)

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    classes = base_for_split.classes  # список WordNet-ID, порядок = порядок виходів моделі
    return train_ds, val_ds, classes


def get_dataloaders(root, batch_size=32, val_fraction=0.15, seed=42, num_workers=4):
    """Обгортає датасети в DataLoader-и, що віддають батчі (x, y):
      x — тензор зображень shape (batch_size, 3, 224, 224), значення в [0, 1];
      y — тензор міток класів shape (batch_size,), цілі числа 0..num_classes-1.
    """
    train_ds, val_ds, classes = get_datasets(root, val_fraction, seed)

    # Генератор з фіксованим seed для перемішування train-набору: порядок
    # батчів буде однаковим між запусками (відтворюваність експериментів).
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,             # нове перемішування на кожній епосі
        num_workers=num_workers,  # паралельні процеси для читання/аугментації зображень
        generator=g,
        drop_last=True,           # відкидаємо останній неповний батч, щоб усі батчі
                                  # мали однаковий розмір (стабільніша статистика BatchNorm)
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False,            # для оцінки порядок не важливий -> детермінований прохід
        num_workers=num_workers,
        # drop_last=False (за замовчуванням): оцінюємо ВСІ валідаційні зображення
    )
    return train_loader, val_loader, classes
