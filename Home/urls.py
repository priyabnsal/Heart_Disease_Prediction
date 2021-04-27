from django.contrib import admin
from django.conf.urls import url
from django.urls import path
from Home import views
urlpatterns = [
    path('', views.index, name='Home'),
    path('about', views.about, name='about'),
    path('predict', views.predict, name='predict'),
]