from django.db import models

# Create your models here.
from django.db import models

class Topic(models.Model):
    name = models.CharField(max_length=255, unique=True, verbose_name="Тема")

    def __str__(self):
        return self.name

class Keyword(models.Model):
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name="keywords")
    word = models.CharField(max_length=255, verbose_name="Ключевое слово")

    def __str__(self):
        return self.word

class Task(models.Model):
    text = models.TextField(verbose_name="Текст задания")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата создания")
    assignee_name = models.CharField(max_length=255, verbose_name="Имя исполнителя", blank=True, null=True)
    assignee_email = models.EmailField(verbose_name="Email исполнителя", blank=True, null=True)

    # Статус задачи (например, 'не обработано', 'обработано', 'в облаке слов')
    status = models.CharField(max_length=50, default='не обработано', verbose_name="Статус задачи")

    def __str__(self):
        return self.text[:50]


class AssignedTask(models.Model):
    task = models.OneToOneField(Task, on_delete=models.CASCADE, verbose_name="Задание")
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, verbose_name="Тема")
    confidence = models.FloatField(verbose_name="Точность соответствия")
    assignee_name = models.CharField(max_length=255, verbose_name="Имя исполнителя", blank=True, null=True)
    assignee_email = models.EmailField(verbose_name="Email исполнителя", blank=True, null=True)

    def __str__(self):
        return f"{self.task.text[:50]} -> {self.topic.name} ({self.confidence:.2f}%)"

