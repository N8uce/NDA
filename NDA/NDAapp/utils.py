import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
from .models import Topic, Keyword, Task, AssignedTask
import logging
from wordcloud import WordCloud
import os

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Функция для обучения модели
def train_model():
    topics = Topic.objects.prefetch_related("keywords").all()

    # Создаем обучающую выборку: темы + ключевые слова
    train_data = []
    train_labels = []

    for topic in topics:
        keywords = [kw.word for kw in topic.keywords.all()]
        if keywords:
            train_data.extend(keywords)
            train_labels.extend([topic.name] * len(keywords))

    if not train_data:
        logger.warning("Нет данных для обучения модели!")
        return None, None, None  # Возвращаем три None, если нет данных

    # Преобразуем метки в числа
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)

    # Токенизация текста
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_data)
    sequences = tokenizer.texts_to_sequences(train_data)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=20)

    # Разделим данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, train_labels, test_size=0.2)

    # Создаем модель нейросети
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=20),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(set(train_labels)), activation='softmax')  # количество классов равно количеству уникальных тем
    ])

    # Используем правильную версию оптимизатора
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Обучаем модель
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32)

    return model, tokenizer, label_encoder  # Возвращаем все три значения


# Функция для предсказания темы задания
def classify_task(task_text, model, tokenizer, label_encoder, threshold=0.9):
    if not model:
        return None, 0  # Если модели нет, ничего не возвращаем

    sequence = tokenizer.texts_to_sequences([task_text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=20)

    # Получаем вероятности для каждой темы
    probabilities = model.predict(padded_sequence)
    max_prob = np.max(probabilities)  # Максимальная вероятность
    predicted_class = np.argmax(probabilities)  # Индекс темы с максимальной вероятностью
    predicted_topic = label_encoder.inverse_transform([predicted_class])[0]  # Преобразуем индекс в тему

    return predicted_topic, max_prob


# Функция для генерации облака слов
def generate_word_cloud():
    tasks = Task.objects.all()  # Получаем все задания
    if not tasks:
        print("⚠️ Нет заданий для обработки")
        return None

    # Собираем слова из всех заданий
    text = " ".join(task.text for task in tasks)

    # Создаем облако слов
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Сохраняем изображение
    img_path = "media/wordcloud.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    wordcloud.to_file(img_path)

    # Собираем данные для слов
    word_data = []
    for word, freq in wordcloud.words_.items():
        word_data.append({
            'word': word,
            'freq': freq,
            'x': random.randint(0, 100),  # случайные координаты для каждого слова
            'y': random.randint(0, 100),
        })

    return img_path, word_data


# Основная функция обработки заданий
def process_tasks():
    model, tokenizer, label_encoder = train_model()
    if not model:
        print("⚠️ Нет данных для обучения модели!")
        return

    # Получаем все задания, которые еще не распределены
    unassigned_tasks = Task.objects.filter(assignedtask__isnull=True)

    for task in unassigned_tasks:
        # Классифицируем задание и получаем тему и уверенность
        predicted_topic, confidence = classify_task(task.text, model, tokenizer, label_encoder)

        if confidence >= 0.9:
            # Находим тему по предсказанному имени
            topic = Topic.objects.filter(name=predicted_topic).first()
            if topic:
                # Создаем запись в AssignedTask
                AssignedTask.objects.create(task=task, topic=topic, confidence=confidence)
                task.delete()  # Удаляем задание из общего списка
                print(f"✅ Задание '{task.text[:30]}...' распределено в тему '{predicted_topic}' (точность {confidence:.2f})")
        else:
            # Если уверенность меньше порога, оставляем задание без изменения
            print(f"⚠️ Задание '{task.text[:30]}...' осталось без темы (точность {confidence:.2f})")

    # Генерируем облако слов после обработки задач
    generate_word_cloud()
