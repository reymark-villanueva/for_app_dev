from pathlib import Path

from django.conf import settings
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.core.exceptions import ValidationError
from django.http import FileResponse, Http404
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.http import require_POST

from mysite.rate_limit import is_rate_limited


username_validator = UnicodeUsernameValidator()


def index(request):
    if request.user.is_authenticated:
        return redirect('scholarships:student_form')
    return render(request, 'home.html')


@require_POST
def login_submit(request):
    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '')

    if is_rate_limited(request, 'login-ip', 30, 300):
        messages.error(request, 'Too many login attempts. Please wait a few minutes.')
        return render(request, 'home.html', status=429)

    if username and is_rate_limited(request, 'login-user', 10, 600, username):
        messages.error(request, 'Too many login attempts. Please wait a few minutes.')
        return render(request, 'home.html', status=429)

    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        return redirect('scholarships:student_form')

    messages.error(request, 'Invalid username or password.')
    return render(request, 'home.html')


def signup(request):
    if request.user.is_authenticated:
        return redirect('scholarships:student_form')
    return render(request, 'signup.html')


@require_POST
def signup_submit(request):
    username = request.POST.get('username', '').strip()
    password = request.POST.get('password', '')

    if is_rate_limited(request, 'signup-ip', 5, 3600):
        messages.error(request, 'Too many signup attempts. Please try again later.')
        return render(request, 'signup.html', status=429)

    if not username or not password:
        messages.error(request, 'Username and password are required.')
        return render(request, 'signup.html')

    try:
        username_validator(username)
        validate_password(password, user=User(username=username))
    except ValidationError as exc:
        messages.error(request, ' '.join(exc.messages))
        return render(request, 'signup.html')

    if len(username) > User._meta.get_field('username').max_length:
        messages.error(request, 'Username is too long.')
        return render(request, 'signup.html')

    if User.objects.filter(username=username).exists():
        messages.error(request, 'Username already taken.')
        return render(request, 'signup.html')

    user = User.objects.create_user(username=username, password=password)
    login(request, user)
    return redirect('scholarships:student_form')


@require_POST
def logout_view(request):
    logout(request)
    request.session.flush()  # Clear session data completely
    return redirect('home')


def logo_srs(request):
    logo_path = Path(settings.BASE_DIR) / "scholarships" / "templates" / "scholarships" / "logo SRS.svg"
    if not logo_path.exists():
        raise Http404("Logo file not found.")
    return FileResponse(logo_path.open("rb"), content_type="image/svg+xml")
