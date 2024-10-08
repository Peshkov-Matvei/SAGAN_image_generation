# SAGAN_image_generation

Этот проект представляет собой реализацию **Self-Attention Generative Adversarial Network (SAGAN)** для генерации изображений с использованием PyTorch. SAGAN использует механизм самовнимания, чтобы улучшить качество сгенерированных изображений, позволяя модели лучше улавливать глобальные зависимости в изображении.

## Описание

SAGAN расширяет стандартные GANs, добавляя механизм самовнимания, что позволяет улучшить генерацию изображений за счёт учёта пространственных зависимостей. В этом проекте применяется Self-Attention в архитектуре генератора и дискриминатора для улучшения реалистичности сгенерированных изображений.

## Структура проекта

- `models/`: Содержит архитектуры моделей генератора и дискриминатора.
- `utils/`: Вспомогательные функции, включая модуль загрузки данных и самовнимания.
- `train.py`: Основной скрипт для обучения моделей.
- `data/`: Папка для хранения датасета CelebA.
- `outputs/`: Папка для сохранения сгенерированных изображений.

## Установка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/yourusername/sagan-image-generation.git
   cd sagan-image-generation
   ```
2. Установите необходимые зависимости:
   ```
   pip install -r requirements.txt
   ```
3. Убедитесь, что папка data содержит все необходимые изображения. Вы можете скачать их из официального сайта CelebA.

## Использование

Для запуска обучения модели используйте команду:

  ```
  python train.py
  ```

Процесс обучения будет выводить информацию о прогрессе в консоль и сохранять сгенерированные изображения в папку outputs/.

## Параметры
- `z_dim`: Размерность вектора шума, подаваемого на вход генератору (по умолчанию 100).
- `image_channels`: Количество каналов в изображениях (по умолчанию 3 для RGB).
- `batch_size`: Размер мини-пакета для обучения (по умолчанию 64).
- `num_epochs`: Количество эпох для обучения (по умолчанию 100).
Эти параметры можно изменить в файле train_sagan.py

## Результаты

В процессе обучения каждые 10 эпох будут сохраняться сгенерированные изображения в папке outputs/. Вы можете просмотреть прогресс и оценить качество генерации на основе этих изображений.

## Автор

Пешков Матвей (https://github.com/Peshkov-Matvei).
