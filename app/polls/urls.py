from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("home/", views.index, name="home"),
    path("signup/", views.signup, name="signup"),
    path("login_submit/", views.login_submit, name="login_submit"),
    path("signup_submit/", views.signup_submit, name="signup_submit"),
    path("logout/", views.logout_view, name="logout"),
]
