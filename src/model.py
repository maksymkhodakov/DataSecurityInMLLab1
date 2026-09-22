"""
Визначення моделі: попередньо натренована ResNet-18 (torchvision), дотренована
(fine-tuned) для 10 класів imagenet-10.

Ключове архітектурне рішення: нормалізація (ImageNet mean/std) винесена
ВСЕРЕДИНУ моделі як перший шар (клас Normalize), а не робиться у трансформах
датасету. Завдяки цьому:
  - датасет (data.py) віддає зображення у "сирому" діапазоні [0, 1];
  - атаки (attacks.py: FGSM, Gaussian Noise) також працюють у [0, 1];
  - значення eps (FGSM) і sigma (Gaussian Noise) мають прямий фізичний сенс
    у термінах пікселів, а не в "перенормованому" внутрішньому просторі
    ознак моделі — інакше eps=8/255 означало б РІЗНУ силу збурення залежно
    від того, до чи після нормалізації воно застосоване.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Стандартні статистики ImageNet (те, на чому тренувались публічні ваги resnet18).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Normalize(nn.Module):
    """Перший шар моделі: (x - mean) / std по кожному з 3 каналів.

    Реалізовано як nn.Module (а не як torchvision-transform), щоб цей крок
    автоматично потрапляв у граф обчислень і differentiable-атаки (FGSM)
    могли коректно рахувати градієнт crossентропії відносно "сирого" входу
    x (у [0,1]), а не відносно вже нормалізованого тензора.
    """
    def __init__(self, mean, std):
        super().__init__()
        # register_buffer — не параметр (не оновлюється оптимізатором), але
        # автоматично переноситься на потрібний пристрій разом з .to(device).
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def build_model(num_classes=10, pretrained=True):
    """Створює модель: Normalize -> ResNet-18 (backbone) -> Linear(num_classes).

    pretrained=True  — завантажує ваги, натреновані на ImageNet-1k (1000
                        класів), і дотреновує їх під нашу задачу (transfer
                        learning; класи imagenet-10 — підмножина WordNet-ID
                        оригінального ImageNet, тому це особливо ефективно).
    pretrained=False — використовується в evaluate.py, де ваги однаково
                        одразу перезаписуються з checkpoint, тож завантажувати
                        їх з інтернету не потрібно (швидше, працює офлайн).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = resnet18(weights=weights)
    # Останній (класифікаційний) шар ResNet-18 має 1000 виходів (класи
    # ImageNet-1k) — замінюємо на num_classes (10 для imagenet-10). Нові
    # ваги цього шару ініціалізуються випадково й тренуються з нуля разом
    # з fine-tuning решти backbone.
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    model = nn.Sequential(
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
        backbone,
    )
    return model
