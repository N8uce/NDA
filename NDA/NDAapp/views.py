from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from collections import Counter
import json, sys, subprocess
from .models import Task, Topic, Keyword

def index(request):
    return render(request, "index.html")


def get_word_cloud(request):
    tasks = Task.objects.filter(status="в облаке слов")  # Статус с маленькими буквами
    if not tasks:
        return JsonResponse({"error": "Нет заданий для облака слов"}, status=400)

    words = []
    for task in tasks:
        words.extend(task.text.split())

    word_counts = Counter(words)
    word_data = [{"text": word, "weight": count} for word, count in word_counts.items()]
    return JsonResponse({"word_data": word_data})



def wordcloud_view(request):
    tasks = Task.objects.all()
    if not tasks:
        return render(request, 'wordcloud.html', {"error": "Нет заданий для отображения облака слов"})
    words = []
    for task in tasks:
        words.extend(task.text.split())
    word_counts = Counter(words)
    word_data = [{"text": word, "weight": count} for word, count in word_counts.items()]
    topics = Topic.objects.all()  # Предполагается, что у Topic есть related_name для keywords
    return render(request, 'wordcloud.html', {"word_data": word_data, "topics": topics})

@csrf_exempt
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
            topic = Topic.objects.create(name=topic_name)

        Keyword.objects.create(topic=topic, word=keyword)
        return JsonResponse({"message": f"Ключевое слово '{keyword}' добавлено в тему '{topic_name}'"})

    return JsonResponse({"error": "Метод не поддерживается"}, status=405)


@csrf_exempt
def update_keyword_topic(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            keyword_text = data.get("keyword")
            new_topic_id = data.get("new_topic_id")
            new_topic_name = data.get("new_topic_name")  # имя новой темы

            # Если ни ID, ни имя новой темы не переданы, возвращаем ошибку
            if not new_topic_id and not new_topic_name:
                return JsonResponse({"success": False, "error": "Тема не указана"}, status=400)

            # Определяем тему: если указан ID существующей темы, используем её,
            # иначе, если передано имя новой темы, пытаемся найти или создать её.
            if new_topic_id:
                topic = Topic.objects.get(id=new_topic_id)
            else:
                topic, _ = Topic.objects.get_or_create(name=new_topic_name)

            # Теперь создаём или получаем Keyword с обязательным указанием темы
            keyword, created = Keyword.objects.get_or_create(
                word=keyword_text,
                defaults={'topic': topic}
            )
            # Если ключевое слово уже существует, обновляем его тему
            if not created:
                keyword.topic = topic
                keyword.save()

            return JsonResponse({"success": True})
        except Topic.DoesNotExist:
            return JsonResponse({"success": False, "error": "Тема не найдена"})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    return JsonResponse({"success": False, "error": "Метод запроса должен быть POST"})





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


@csrf_exempt
def run_task(request):
    if request.method == "POST":
        try:
            python_executable = sys.executable
            subprocess.run([python_executable, "manage.py", "process_tasks"], check=True)

            # Получение списка слов для облака
            tasks = Task.objects.filter(status="в облаке слов")
            words = []
            for task in tasks:
                words.extend(task.text.split())

            word_counts = Counter(words)
            word_data = [{"text": word, "weight": count} for word, count in word_counts.items()]

            return JsonResponse({"success": True, "words": word_data})

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

    return JsonResponse({"success": False, "error": "Invalid request"})

