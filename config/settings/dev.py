import os

from .base import *  # noqa: F403, F401

# Helper functions to replace decouple functionality


def get_bool_env(key, default=False):
    """Convert environment variable to boolean"""
    value = os.environ.get(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_int_env(key, default=0):
    """Convert environment variable to integer"""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def get_csv_env(key, default=""):
    """Convert comma-separated environment variable to list"""
    value = os.environ.get(key, default)
    return [item.strip() for item in value.split(",") if item.strip()]


# Settings
DEBUG = get_bool_env("DEBUG", True)

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

CORS_ORIGIN_ALLOW_ALL = get_bool_env("CORS_ORIGIN_ALLOW_ALL", False)

CORS_ALLOWED_ORIGINS = get_csv_env(
    "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
)

CSRF_TRUSTED_ORIGINS = get_csv_env(
    "CSRF_TRUSTED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
)

EMAIL_BACKEND = os.environ.get(
    "EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend"
)
EMAIL_HOST = os.environ.get("EMAIL_HOST", "")
EMAIL_PORT = get_int_env("EMAIL_PORT", 587)
EMAIL_USE_TLS = get_bool_env("EMAIL_USE_TLS", True)
EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "")

DEFAULT_FROM_EMAIL = os.environ.get("DEFAULT_FROM_EMAIL", "no-reply@example.com")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
