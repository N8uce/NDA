from django.core.management.base import BaseCommand
from NDAapp.utils import process_tasks

class Command(BaseCommand):
    help = "Распределяет задания по темам"

    def handle(self, *args, **kwargs):
        process_tasks()
        self.stdout.write(self.style.SUCCESS("✅ Обработка завершена"))
