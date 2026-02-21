from django.shortcuts import render
from django.http import HttpResponse
from polls import views

def index(request):
    return render(request, 'home.html')

def base(request):
    return render(request, "base.html")

def login(request):
    return render(request, "login.html")

def signup(request):
    return render(request, "signup.html")



