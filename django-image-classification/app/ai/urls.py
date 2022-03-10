from django.urls import path
from . import views

urlpatterns = [
    path('home/',views.index,name='index'),
    path('predict/',views.predictImage,name='predict'),
    path('',views.viewDataBase,name='gallery'),
]