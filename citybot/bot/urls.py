from django.contrib import admin
from django.urls import path
from bot import views

urlpatterns = [
    path("", views.index, name='index'),
    path("home", views.home, name='home'),
    path("bot", views.bot, name='bot')
]
