# Generated by Django 3.0 on 2021-12-03 12:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('EVTools', '0022_auto_20211202_1258'),
    ]

    operations = [
        migrations.AlterField(
            model_name='vehiclecategorymodel',
            name='e',
            field=models.FloatField(blank=True, default=0.0, null=True),
        ),
    ]
