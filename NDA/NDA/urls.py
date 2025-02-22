"""
URL configuration for NDA project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.contrib import admin
from django.conf.urls.static import static
from django.urls import path
from NDAapp import views

urlpatterns = [
    path('', views.index, name='index'),  # Главная страница
    path("update_keyword_topic/", views.update_keyword_topic, name="update_keyword_topic"),
    path("admin/", admin.site.urls),
    path('wordcloud/', views.wordcloud_view, name='wordcloud'),
    path("add_keyword/", views.add_keyword, name="add_keyword"),
    path('api/get_topics/', views.get_topics, name='get_topics'),
    path('update_keyword_topic/', views.update_keyword_topic, name='update_keyword_topic'),
    path("run_task/", views.run_task, name="run_task"),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
