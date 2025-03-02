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
    for epoch in range(900):  # Обучаем 1000 эпох
        optimizer.zero_grad()

        outputs = model(X_train, labels=y_train)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{900}, Loss: {loss.item()}")

    # Сохраняем модель и токенизатор
    os.makedirs(MODEL_PATH, exist_ok=True)  # Убедимся, что директория существует
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    label_encoder_path = os.path.join(MODEL_PATH, 'label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    return model, tokenizer, label_encoder


# Функция для загрузки модели
# Функция для загрузки модели
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, 'config.json')):
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



from django.core.mail import send_mail
from django.conf import settings

def send_task_email(assignee_name, assignee_email, task_text, predicted_topic, confidence):
    subject = f'Задание распределено по теме "{predicted_topic}"'
    message = f'Здравствуйте, {assignee_name}!\n\nВаше задание: "{task_text[:50]}..." было распределено по теме "{predicted_topic}" с точностью {confidence:.2f}.\n\nПожалуйста, приступайте к выполнению задания.'
    from_email = settings.DEFAULT_FROM_EMAIL

    try:
        send_mail(subject, message, from_email, [assignee_email])
        print(f"✅ Письмо отправлено на {assignee_email}")
    except Exception as e:
        print(f"⚠️ Ошибка при отправке письма: {e}")


# Основная функция обработки заданий
# Изменение в process_tasks
def process_tasks():
    model, tokenizer, label_encoder = load_model()
    if not model:
        print("⚠️ Нет данных для загрузки модели!")
        return

    # Получаем все задания, которые еще не обработаны (статус 'не обработано')
    unassigned_tasks = Task.objects.filter(status__in=["не обработано", "в облаке слов"])

    for task in unassigned_tasks:
        # Классифицируем задание и получаем тему и уверенность
        predicted_topic, confidence = classify_task(task.text, model, tokenizer, label_encoder)

        if confidence >= 0.7:
            # Находим тему по предсказанному имени
            topic = Topic.objects.filter(name=predicted_topic).first()
            if topic:
                # Создаем запись в AssignedTask с дополнительными полями
                AssignedTask.objects.create(
                    task=task,
                    topic=topic,
                    confidence=confidence,
                    assignee_name=task.assignee_name,  # Добавляем имя исполнителя
                    assignee_email=task.assignee_email  # Добавляем email исполнителя
                )

                # Обновляем статус задачи на 'обработано'
                task.status = 'обработано'
                task.save()  # Сохраняем изменения

                print(
                    f"✅ Задание '{task.text[:30]}...' распределено в тему '{predicted_topic}' (точность {confidence:.2f})")

                # Записываем задание в файл
                write_to_file(task.text, predicted_topic, confidence)
                send_task_email(task.assignee_name, task.assignee_email, task.text, predicted_topic, confidence)

        else:
            # Если уверенность меньше порога, задание отправляется в облако слов
            task.status = 'в облаке слов'
            task.keywords_sent_to_wordcloud = True
            task.save()  # Сохраняем изменения

            print(f"⚠️ Задание '{task.text[:30]}...' отправлено в облако слов (точность {confidence:.2f})")



