from decouple import config

from .base import *  # noqa: F403

# Custom CSV parser function to replace Csv class


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


# SECURITY
DEBUG = config("DEBUG", default=False, cast=bool)
ALLOWED_HOSTS = parse_csv(config("ALLOWED_HOSTS", default=""))

# CORS & CSRF
CORS_ORIGIN_ALLOW_ALL = config("CORS_ORIGIN_ALLOW_ALL", default=False, cast=bool)
CORS_ALLOWED_ORIGINS = parse_csv(config("CORS_ALLOWED_ORIGINS", default=""))
CSRF_TRUSTED_ORIGINS = parse_csv(config("CSRF_TRUSTED_ORIGINS", default=""))

# EMAIL
EMAIL_BACKEND = config("EMAIL_BACKEND")
EMAIL_HOST = config("EMAIL_HOST")
EMAIL_PORT = config("EMAIL_PORT", cast=int)
EMAIL_USE_TLS = config("EMAIL_USE_TLS", cast=bool)
EMAIL_HOST_USER = config("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD")
DEFAULT_FROM_EMAIL = config("DEFAULT_FROM_EMAIL")

# Frontend
FRONTEND_URL = config("FRONTEND_URL")
