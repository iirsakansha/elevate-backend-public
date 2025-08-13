#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    # Check if we're in production environment
    # Priority: Environment variable > .env file > default to dev

    # First check if DJANGO_SETTINGS_MODULE is already set
    if not os.environ.get("DJANGO_SETTINGS_MODULE"):
        # Try to load from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

    # Set default based on environment or fall back to dev
    default_settings = os.environ.get("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", default_settings)

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
