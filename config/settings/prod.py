import os

from dotenv import load_dotenv

from .base import *

# Load .env file
env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"
)
load_dotenv(env_path)


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


# SECURITY
DEBUG = get_bool_env("DEBUG", False)
ALLOWED_HOSTS = get_csv_env("ALLOWED_HOSTS", "")

# DATABASE CONFIGURATION
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME"),
        "USER": os.environ.get("DB_USER"),
        "PASSWORD": os.environ.get("DB_PASSWORD"),
        "HOST": os.environ.get("DB_HOST"),
        "PORT": get_int_env("DB_PORT", 5432),
        "OPTIONS": {
            "sslmode": "require",  # Required for AWS RDS
        },
    }
}

# CORS & CSRF
CORS_ORIGIN_ALLOW_ALL = get_bool_env("CORS_ORIGIN_ALLOW_ALL", False)
CORS_ALLOWED_ORIGINS = get_csv_env("CORS_ALLOWED_ORIGINS", "")
CSRF_TRUSTED_ORIGINS = get_csv_env("CSRF_TRUSTED_ORIGINS", "")

# EMAIL CONFIGURATION
EMAIL_BACKEND = os.environ.get(
    "EMAIL_BACKEND", "django.core.mail.backends.smtp.EmailBackend"
)
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = get_int_env("EMAIL_PORT", 587)
EMAIL_USE_TLS = get_bool_env("EMAIL_USE_TLS", True)
EMAIL_USE_SSL = get_bool_env("EMAIL_USE_SSL", False)
EMAIL_HOST_USER = os.environ.get("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.environ.get("EMAIL_HOST_PASSWORD", "")
EMAIL_TIMEOUT = get_int_env("EMAIL_TIMEOUT", 60)
DEFAULT_FROM_EMAIL = os.environ.get("DEFAULT_FROM_EMAIL", EMAIL_HOST_USER)
SERVER_EMAIL = os.environ.get("SERVER_EMAIL", DEFAULT_FROM_EMAIL)
FRONTEND_URL = os.environ.get("FRONTEND_URL", "")
