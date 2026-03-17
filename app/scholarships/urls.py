from django.urls import path

from . import views

app_name = 'scholarships'

urlpatterns = [
    path('apply/', views.student_form_view, name='student_form'),
    path('results/<int:profile_id>/', views.results_view, name='results'),
    path('history/', views.history_view, name='history'),
]
