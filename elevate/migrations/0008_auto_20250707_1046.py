# elevate/migrations/000X_auto_YYYYMMDD_HHMM.py
from django.db import migrations


def migrate_categories(apps, schema_editor):
    Analysis = apps.get_model('elevate', 'Analysis')
    for analysis in Analysis.objects.all():
        load_categories = []
        for i in range(1, 7):
            field = f'loadCategory{i}'
            if hasattr(analysis, field) and getattr(analysis, field):
                load_categories.append(getattr(analysis, field))
        analysis.load_categories.set(load_categories)

        vehicle_categories = []
        for i in range(1, 6):
            field = f'vehicleCategoryData{i}'
            if hasattr(analysis, field) and getattr(analysis, field):
                vehicle_categories.append(getattr(analysis, field))
        analysis.vehicle_categories.set(vehicle_categories)


class Migration(migrations.Migration):
    dependencies = [
        ('elevate', '0007_alter_analysis_options_alter_files_options_and_more'),
    ]

    operations = [
        migrations.RunPython(migrate_categories),
    ]
