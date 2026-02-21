from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("polls/", views.index, name="polls_index"),
    path("base/", views.base, name="base"),
    path("home/", views.index, name="home"),
    path("signup/", views.signup, name="signup"),
]