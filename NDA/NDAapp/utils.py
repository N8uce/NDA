import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .models import Topic, Task, AssignedTask
import logging
import pickle
import csv

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Путь к директории, где сохраняется модель
MODEL_PATH = './model'
ASSIGNED_TASKS_FILE = './assigned_tasks.csv'  # Путь к CSV файлу для сохранения заданий


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
    tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')

    train_encodings = tokenizer(train_data, truncation=True, padding=True, return_tensors='pt')

    # Разделим данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(train_encodings['input_ids'], train_labels, test_size=0.2)

    # Создаем модель
    model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2',
                                                          num_labels=len(set(train_labels)))

    # Обучаем модель
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Преобразуем данные в тензоры
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    # Обучение
    model.train()
    for epoch in range(1000):  # Обучаем 1000 эпох
        optimizer.zero_grad()

        outputs = model(X_train, labels=y_train)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{1000}, Loss: {loss.item()}")

    # Сохраняем модель и токенизатор
    os.makedirs(MODEL_PATH, exist_ok=True)  # Убедимся, что директория существует
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    label_encoder_path = os.path.join(MODEL_PATH, 'label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    return model, tokenizer, label_encoder


# Функция для загрузки модели
def load_model():
    if os.path.exists(MODEL_PATH):
        # Загружаем модель и токенизатор
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

        # Загружаем label encoder
        label_encoder_path = os.path.join(MODEL_PATH, 'label_encoder.pkl')
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        return model, tokenizer, label_encoder
    else:
        print("⚠️ Модель не найдена, начинаем обучение заново.")
        return train_model()  # Если модель не найдена, начинаем обучение заново


# Функция для предсказания темы задания
def classify_task(task_text, model, tokenizer, label_encoder, threshold=0.9):
    if not model:
        return None, 0

    inputs = tokenizer(task_text, return_tensors="pt", truncation=True, padding=True, max_length=20)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    max_prob = torch.max(probabilities).item()  # макс вероятность
    predicted_class = torch.argmax(probabilities).item()  # Индекс темы с макс вероятностью
    predicted_topic = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_topic, max_prob


# Функция для записи задания в файл
def write_to_file(task_text, predicted_topic, confidence):
    with open(ASSIGNED_TASKS_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([task_text, predicted_topic, confidence])
        print(f"✅ Задание записано в файл: '{task_text[:30]}...' в тему '{predicted_topic}' (точность {confidence:.2f})")


# Основная функция обработки заданий
def process_tasks():
    model, tokenizer, label_encoder = load_model()
    if not model:
        print("⚠️ Нет данных для загрузки модели!")
        return

    # Получаем все задания, которые еще не распределены
    unassigned_tasks = Task.objects.filter(assignedtask__isnull=True)

    for task in unassigned_tasks:
        # Классифицируем задание и получаем тему и уверенность
        predicted_topic, confidence = classify_task(task.text, model, tokenizer, label_encoder)

        if confidence >= 0.6:
            # Находим тему по предсказанному имени
            topic = Topic.objects.filter(name=predicted_topic).first()
            if topic:
                # Создаем запись в AssignedTask
                AssignedTask.objects.create(task=task, topic=topic, confidence=confidence)
                task.delete()  # Удаляем задание из общего списка
                print(f"✅ Задание '{task.text[:30]}...' распределено в тему '{predicted_topic}' (точность {confidence:.2f})")

                # Записываем задание в файл
                write_to_file(task.text, predicted_topic, confidence)
        else:
            # Если уверенность меньше порога, оставляем задание без изменения
            print(f"⚠️ Задание '{task.text[:30]}...' осталось без темы (точность {confidence:.2f})")
