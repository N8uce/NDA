from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Topic, Keyword, Task, AssignedTask

class KeywordInline(admin.TabularInline):
    model = Keyword
    extra = 1  # Количество пустых строк для добавления ключевых слов

class TopicAdmin(admin.ModelAdmin):
    inlines = [KeywordInline]  # Вставляем форму для добавления ключевых слов прямо в форму темы

class AssignedTaskAdmin(admin.ModelAdmin):
    list_display = ('task', 'topic', 'confidence')  # Поля, которые будут отображаться в списке

class TaskAdmin(admin.ModelAdmin):
    list_display = ('text', 'created_at')  # Поля для заданий

# Регистрируем модели
admin.site.register(Topic, TopicAdmin)
admin.site.register(Keyword)
admin.site.register(Task, TaskAdmin)
admin.site.register(AssignedTask, AssignedTaskAdmin)
