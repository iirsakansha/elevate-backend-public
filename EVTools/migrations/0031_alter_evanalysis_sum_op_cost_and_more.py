# Generated by Django 5.2 on 2025-04-22 09:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        (
            "EVTools",
            "0030_alter_evanalysis_options_alter_useranalysis_options_and_more",
        ),
    ]

    operations = [
        migrations.AlterField(
            model_name="evanalysis",
            name="sum_op_cost",
            field=models.FloatField(blank=True, default=0, null=True),
        ),
        migrations.AlterField(
            model_name="evanalysis",
            name="sum_pk_cost",
            field=models.FloatField(blank=True, default=0, null=True),
        ),
        migrations.AlterField(
            model_name="evanalysis",
            name="sum_zero_cost",
            field=models.FloatField(blank=True, default=0, null=True),
        ),
        migrations.AlterField(
            model_name="evanalysis",
            name="win_op_cost",
            field=models.FloatField(blank=True, default=0, null=True),
        ),
        migrations.AlterField(
            model_name="evanalysis",
            name="win_pk_cost",
            field=models.FloatField(blank=True, default=0, null=True),
        ),
    ]
