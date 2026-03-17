from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages


def index(request):
    if request.user.is_authenticated:
        return redirect('scholarships:student_form')
    return render(request, 'home.html')


def login_submit(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('scholarships:student_form')
        else:
            messages.error(request, 'Invalid username or password.')
            return render(request, 'home.html')
    return redirect('home')


def signup(request):
    if request.user.is_authenticated:
        return redirect('scholarships:student_form')
    return render(request, 'signup.html')


def signup_submit(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if not username or not password:
            messages.error(request, 'Username and password are required.')
            return render(request, 'signup.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
            return render(request, 'signup.html')

        user = User.objects.create_user(username=username, password=password)
        login(request, user)
        return redirect('scholarships:student_form')
    return redirect('signup')


def logout_view(request):
    logout(request)
    return redirect('home')
