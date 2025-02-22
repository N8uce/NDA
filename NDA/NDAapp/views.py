from django.shortcuts import render

# Create your views here.



def index(request):
    return render(request, "index.html")


def get_word_cloud(request):
    tasks = Task.objects.all()  # Предполагаем, что у вас есть модель Task
    if not tasks:
        return JsonResponse({"error": "Нет заданий для облака слов"}, status=400)

    # Собираем все слова из текста заданий
    words = []
    for task in tasks:
        words.extend(task.text.split())

    # Подсчитываем частоту появления слов
    word_counts = Counter(words)

    # Преобразуем в формат для wordcloud.js
    word_data = [{"text": word, "weight": count} for word, count in word_counts.items()]

    return JsonResponse({"word_data": word_data})




from django.shortcuts import render
from .models import Task
from collections import Counter
import json

def wordcloud_view(request):
    tasks = Task.objects.all()  # Получаем все задачи
    if not tasks:
        return render(request, 'wordcloud.html', {"error": "Нет заданий для отображения облака слов"})

    # Собираем все слова из текста заданий
    words = []
    for task in tasks:
        words.extend(task.text.split())

    # Подсчитываем частоту появления слов
    word_counts = Counter(words)

    # Преобразуем в формат для wordcloud.js
    word_data = [{"text": word, "weight": count} for word, count in word_counts.items()]

    return render(request, 'wordcloud.html', {"word_data": word_data})

from django.views.decorators.csrf import csrf_exempt


def get_topics(request):
    topics = Topic.objects.all()
    topics_data = [{"name": topic.name} for topic in topics]
    return JsonResponse({"topics": topics_data})


@csrf_exempt
def add_keyword(request):
    if request.method == "POST":
        data = json.loads(request.body)
        topic_name = data.get("topic")
        keyword = data.get("keyword")

        topic = Topic.objects.filter(name=topic_name).first()
        if not topic:
            # Создаем новую тему, если такой нет
            topic = Topic.objects.create(name=topic_name)

        Keyword.objects.create(topic=topic, word=keyword)
        return JsonResponse({"message": f"Ключевое слово '{keyword}' добавлено в тему '{topic_name}'"})

    return JsonResponse({"error": "Метод не поддерживается"}, status=405)



from .models import Topic, Keyword, Task
from .utils import process_tasks  # Функция повторной обработки заданий


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import Keyword, Topic

@csrf_exempt
def update_keyword_topic(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            keyword_text = data.get("keyword")
            new_topic_id = data.get("new_topic_id")

            keyword = Keyword.objects.get(word=keyword_text)
            new_topic = Topic.objects.get(id=new_topic_id)

            keyword.topic = new_topic
            keyword.save()

            return JsonResponse({"success": True})
        except Keyword.DoesNotExist:
            return JsonResponse({"success": False, "error": "Ключевое слово не найдено"})
        except Topic.DoesNotExist:
            return JsonResponse({"success": False, "error": "Тема не найдена"})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

    return JsonResponse({"success": False, "error": "Метод запроса должен быть POST"})



from django.http import JsonResponse
from .models import Topic

@csrf_exempt
def add_topic(request):
    if request.method == "POST":
        data = json.loads(request.body)
        topic_name = data.get("topic_name")

        if not topic_name:
            return JsonResponse({"success": False, "error": "Название темы не может быть пустым!"})

        topic, created = Topic.objects.get_or_create(name=topic_name)
        return JsonResponse({"success": True, "topic_id": topic.id})

    return JsonResponse({"success": False, "error": "Метод не поддерживается"}, status=405)


import subprocess
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import subprocess
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import subprocess
import sys
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def run_task(request):
    if request.method == "POST":
        try:
            python_executable = sys.executable  # Этот путь гарантированно будет использовать правильный Python
            subprocess.Popen([python_executable, "manage.py", "process_tasks"])
            return JsonResponse({"success": True})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

    return JsonResponse({"success": False, "error": "Invalid request"})


