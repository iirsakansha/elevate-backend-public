# Generated by Django 3.0 on 2021-11-18 10:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('EVTools', '0005_auto_20211118_1551'),
    ]

    operations = [
        migrations.RenameField(
            model_name='evanalysis',
            old_name='othes',
            new_name='others',
        ),
        migrations.RenameField(
            model_name='evanalysis',
            old_name='senario',
            new_name='scenario',
        ),
        migrations.RenameField(
            model_name='evanalysis',
            old_name='vahicalCategory',
            new_name='vehicleCategory',
        ),
    ]
