import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("elevate", "0008_auto_20250707_1046"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name="Files",
            name="created_at",
            field=models.DateTimeField(
                auto_now_add=True,
                default=django.utils.timezone.now,  # Use current timestamp as default
                verbose_name="created at",
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="Files",
            name="user",
            field=models.ForeignKey(
                default=1,  # Replace with a valid User ID or a callable
                on_delete=django.db.models.deletion.CASCADE,
                related_name="uploaded_files",
                to=settings.AUTH_USER_MODEL,
                verbose_name="user",
            ),
            preserve_default=False,
        ),
    ]
