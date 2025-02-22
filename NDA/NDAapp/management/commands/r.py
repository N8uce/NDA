import csv

# Данные для файла A (темы и ключевые слова)
topics_data = {
    "Программирование на Python": ["python", "класс", "функция", "модуль", "пакет"],
    "Машинное обучение": ["машинное обучение", "нейронные сети", "регрессия", "классификация", "скрипт"],
    "Разработка веб-приложений": ["веб-разработка", "Django", "Flask", "HTML", "CSS"],
    "Анализ данных": ["анализ данных", "SQL", "пандас", "numpy", "сводная таблица"],
    "Обработка естественного языка": ["NLP", "токенизация", "стемминг", "обработка текста", "текстовый анализ"],
}

# Данные для файла B (задания)
tasks_data = [
    "Напишите программу на Python, которая использует классы и функции для обработки данных.",
    "Разработайте веб-приложение с использованием Django для учета пользователей.",
    "Исследуйте влияние различных алгоритмов классификации на точность предсказаний в машинном обучении.",
    "Используйте SQL для создания базы данных, которая будет хранить информацию о сотрудниках компании.",
    "Реализуйте токенизацию текста с использованием библиотеки NLTK для анализа отзывов пользователей.",
]

# Создание файла A
with open('file_a.csv', mode='w', newline='', encoding='utf-8') as file_a:
    writer = csv.writer(file_a)
    writer.writerow(['Тема', 'Ключевое слово 1', 'Ключевое слово 2', 'Ключевое слово 3', 'Ключевое слово 4', 'Ключевое слово 5'])
    for topic_name, keywords in topics_data.items():
        writer.writerow([topic_name] + keywords)

print("✅ Файл A (темы и ключевые слова) создан!")

# Создание файла B
with open('file_b.csv', mode='w', newline='', encoding='utf-8') as file_b:
    writer = csv.writer(file_b)
    writer.writerow(['Задание'])
    for task_text in tasks_data:
        writer.writerow([task_text])

print("✅ Файл B (задания) создан!")
