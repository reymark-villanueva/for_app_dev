from django.apps import AppConfig
from django.conf import settings


class ScholarshipsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'scholarships'

    def ready(self):
        if getattr(settings, 'PRELOAD_ML_MODELS', False):
            from .ml.engine import load_models
            load_models()
