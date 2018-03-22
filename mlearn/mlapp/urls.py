from django.conf.urls import url
from django.contrib import admin
from mlapp import views

app_name = 'mlapp'

urlpatterns = [
    url(r'^upload/',views.upload.as_view(), name = 'upload'),
    url(r'^$',views.index.as_view(), name = 'index'),
    url(r'^preprocessing/',views.preprocessing.as_view(), name = 'preprocessing'),
    url(r'^modelling/',views.modelling.as_view(), name = 'modelling'),
]
