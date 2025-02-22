import csv
import os
from django.core.management.base import BaseCommand
from NDAapp.models import Topic, Keyword, Task


class Command(BaseCommand):
    help = 'Загружает данные из файлов A (темы и ключевые слова) и B (задания)'

    def handle(self, *args, **kwargs):
        # Получение абсолютного пути к текущей директории скрипта
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Путь к файлам A и B
        file_a_path = os.path.join(base_dir, 'file_a.csv')
        file_b_path = os.path.join(base_dir, 'file_b.csv')

        # Загрузка данных для файла A (темы и ключевые слова)
        with open(file_a_path, mode='r', encoding='utf-8') as file_a:
            reader = csv.reader(file_a)
            for row in reader:
                topic_name = row[0]

                # Проверка, существует ли уже такая тема
                topic, created = Topic.objects.get_or_create(name=topic_name)

                # Добавление ключевых слов, если они еще не существуют
                for keyword in row[1:]:
                    Keyword.objects.get_or_create(word=keyword, topic=topic)

        self.stdout.write(self.style.SUCCESS("✅ Данные для файла A загружены!"))

        # Загрузка данных для файла B (задания)
        with open(file_b_path, mode='r', encoding='utf-8') as file_b:
            reader = csv.reader(file_b)
            for row in reader:
                task_text = row[0]

                # Проверка, существует ли уже такое задание
                if not Task.objects.filter(text=task_text).exists():
                    Task.objects.create(text=task_text)

        self.stdout.write(self.style.SUCCESS("✅ Данные для файла B загружены!"))
