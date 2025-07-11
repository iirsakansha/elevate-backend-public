# Generated by Django 3.0 on 2021-11-23 11:24

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('EVTools', '0009_delete_evanalysis'),
    ]

    operations = [
        migrations.CreateModel(
            name='LoadCategoryModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('category', models.CharField(choices=[('commercial', 'commercial'), ('agriculture', 'agriculture'), ('industrial', 'industrial'), ('residential', 'residential'), ('public', 'public'), ('others', 'others')], default='', max_length=20)),
                ('categoryFile', models.FileField(blank=True, default='', null=True, upload_to='CategotuFileUpload/')),
                ('salesCAGR', models.IntegerField(blank=True, default='0', null=True)),
                ('baseElectricity', models.FloatField(blank=True, default='0', validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)])),
            ],
        ),
        migrations.CreateModel(
            name='vehicleCategoryModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vehicleCategory', models.CharField(choices=[('commercial', 'commercial'), ('agriculture', 'agriculture'), ('industrial', 'industrial'), ('residential', 'residential'), ('public', 'public'), ('others', 'others')], default='', max_length=20)),
                ('n', models.IntegerField(blank=True, default='0', null=True)),
                ('f', models.IntegerField(blank=True, default='0', null=True)),
                ('c', models.IntegerField(blank=True, default='0', null=True)),
                ('p', models.IntegerField(blank=True, default='0', null=True)),
                ('e', models.IntegerField(blank=True, default='0', null=True)),
                ('r', models.IntegerField(blank=True, default='0', null=True)),
                ('k', models.IntegerField(blank=True, default='0', null=True)),
                ('l', models.IntegerField(blank=True, default='0', null=True)),
                ('g', models.IntegerField(blank=True, default='0', null=True)),
                ('h', models.IntegerField(blank=True, default='0', null=True)),
                ('s', models.IntegerField(blank=True, default='0', null=True)),
                ('u', models.IntegerField(blank=True, default='0', null=True)),
                ('CAGR_V', models.IntegerField(blank=True, default='0', null=True)),
            ],
        ),
        migrations.CreateModel(
            name='evAnalysis',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('loadCategory', models.IntegerField(default='0', validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(6)])),
                ('isLoadSplit', models.CharField(choices=[('Yes', 'Yes'), ('No', 'No')], default='', max_length=20)),
                ('isLoadSplitFile', models.FileField(blank=True, default='', null=True, upload_to='LoadSplitFile')),
                ('numOfvehicleCategory', models.IntegerField(blank=True, default='0', null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(5)])),
                ('module', models.CharField(choices=[('managed', 'managed'), ('unmanaged', 'unmanaged')], default='', max_length=20)),
                ('resolution', models.IntegerField(blank=True, default='0', null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)])),
                ('BR_F', models.IntegerField(blank=True, default='0', null=True)),
                ('moduleCategory', models.CharField(choices=[('managed', 'managed'), ('unmanaged', 'unmanaged')], default='', max_length=20)),
                ('isManagedFile', models.FileField(blank=True, default='', null=True, upload_to='ManagedFile/')),
                ('scenario', models.CharField(choices=[('0', 'base + ev'), ('1', 'ev')], default='', max_length=20)),
                ('isOverShot', models.CharField(choices=[('0', 'offshot to be spread over'), ('1', 'per of total area to be spread over')], default='', max_length=20)),
                ('perOfShotToBeSpread', models.FloatField(blank=True, default='0', validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)])),
                ('perOfTotalArea', models.FloatField(blank=True, default='0', validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)])),
                ('loadCategory1', models.ForeignKey(blank=True, max_length=30, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='load_Cat_1', to='EVTools.LoadCategoryModel')),
                ('loadCategory2', models.ForeignKey(blank=True, max_length=30, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='load_Cat_2', to='EVTools.LoadCategoryModel')),
                ('loadCategory3', models.ForeignKey(blank=True, max_length=30, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='load_Cat_3', to='EVTools.LoadCategoryModel')),
                ('loadCategory4', models.ForeignKey(blank=True, max_length=30, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='load_Cat_4', to='EVTools.LoadCategoryModel')),
                ('loadCategory5', models.ForeignKey(blank=True, max_length=30, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='load_Cat_5', to='EVTools.LoadCategoryModel')),
                ('loadCategory6', models.ForeignKey(blank=True, max_length=30, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='load_Cat_6', to='EVTools.LoadCategoryModel')),
                ('vehicleCategoryData1', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='vehicle_Cat_1', to='EVTools.vehicleCategoryModel')),
                ('vehicleCategoryData2', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='vehicle_Cat_2', to='EVTools.vehicleCategoryModel')),
                ('vehicleCategoryData3', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='vehicle_Cat_3', to='EVTools.vehicleCategoryModel')),
                ('vehicleCategoryData4', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='vehicle_Cat_4', to='EVTools.vehicleCategoryModel')),
                ('vehicleCategoryData5', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='vehicle_Cat_5', to='EVTools.vehicleCategoryModel')),
            ],
        ),
    ]
