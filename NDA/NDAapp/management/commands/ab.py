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

        # Загрузка данных для файла B (задания с исполнителями)
        with open(file_b_path, mode='r', encoding='utf-8') as file_b:
            reader = csv.reader(file_b)
            for row in reader:
                if len(row) < 3:
                    self.stdout.write(self.style.WARNING(f"⚠️ Пропущена строка (недостаточно данных): {row}"))
                    continue

                task_text, assignee_name, assignee_email = row

                # Проверка, существует ли уже такое задание
                task, created = Task.objects.get_or_create(
                    text=task_text,
                    defaults={"assignee_name": assignee_name, "assignee_email": assignee_email}
                )

                # Если задание уже существует, но у него нет имени или email, обновляем их
                if not created and (not task.assignee_name or not task.assignee_email):
                    task.assignee_name = assignee_name
                    task.assignee_email = assignee_email
                    task.save()

        self.stdout.write(self.style.SUCCESS("✅ Данные для файла B загружены!"))
