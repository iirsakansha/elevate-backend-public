from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
import json

PERCENTAGE_VALIDATOR = [MinValueValidator(0), MaxValueValidator(100)]


class UserProfile(models.Model):
    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name='profile')
    organization = models.CharField(max_length=255, default='Not assigned')
    invitation_status = models.CharField(max_length=50, default='Pending')
    invitation_username = models.CharField(max_length=255, blank=True)
    temporary_password = models.CharField(max_length=255, blank=True)
    invitation_token = models.CharField(
        max_length=255, unique=True, blank=True)
    password_reset_token = models.CharField(
        max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    role = models.CharField(max_length=50, default='user')

    def __str__(self):
        return f"Profile for {self.user.username}"


class LoadCategoryModel(models.Model):
    CATEGORY_CHOICES = (
        ("commercial", 'Commercial'),
        ("agricultural", 'Agricultural'),
        ("industrial", 'Industrial'),
        ("residential", 'Residential'),
        ("public", 'Public'),
        ("others", 'Others'),
    )
    category = models.CharField(
        max_length=20, choices=CATEGORY_CHOICES, default="")
    categoryFile = models.CharField(
        default="", blank=True, null=True, max_length=400)
    salesCAGR = models.IntegerField(blank=True, null=True, default=0)
    specifySplit = models.FloatField(null=True, blank=True, default=0)

    def __str__(self):
        return self.category


class VehicleCategoryModel(models.Model):
    VEHICLE_CATEGORY_CHOICES = (
        ("car", 'Car'),
        ("bus", 'Bus'),
        ("2-wheeler", '2-Wheeler'),
        ("3-wheeler", '3-Wheeler'),
        ("others", 'Others'),
    )
    vehicleCategory = models.CharField(
        max_length=20, choices=VEHICLE_CATEGORY_CHOICES, default="")
    n = models.IntegerField(default=0, blank=True, null=True)
    f = models.IntegerField(default=0, blank=True, null=True)
    c = models.IntegerField(default=0, blank=True, null=True)
    p = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    e = models.FloatField(default=0.0, blank=True, null=True)
    r = models.IntegerField(default=0, blank=True, null=True)
    k = models.IntegerField(default=0, blank=True, null=True)
    l = models.IntegerField(default=0, blank=True, null=True)
    g = models.IntegerField(default=0, blank=True, null=True)
    h = models.IntegerField(default=0, blank=True, null=True)
    s = models.IntegerField(default=0, blank=True, null=True)
    u = models.IntegerField(default=0, blank=True, null=True)
    rowlimit_xl = models.IntegerField(default=2000000, blank=False, null=False)
    CAGR_V = models.IntegerField(default=0, blank=True, null=True)
    baseElectricityTariff = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)

    def __str__(self):
        return self.vehicleCategory


class Analysis(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='analyses')
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    # Load Category Fields
    loadCategory = models.IntegerField(
        default=0, validators=[MinValueValidator(1), MaxValueValidator(6)])
    isLoadSplit = models.CharField(max_length=20, choices=(
        ("yes", 'Yes'), ("no", 'No')), default="")
    isLoadSplitFile = models.CharField(
        default="", blank=True, null=True, max_length=400)
    loadCategory1 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='load_cat_1')
    loadCategory2 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='load_cat_2')
    loadCategory3 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='load_cat_3')
    loadCategory4 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='load_cat_4')
    loadCategory5 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='load_cat_5')
    loadCategory6 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='load_cat_6')
    category_data = models.JSONField(default=list)

    # Vehicle Category Fields
    numOfvehicleCategory = models.IntegerField(blank=True, null=True, default=0, validators=[
                                               MinValueValidator(1), MaxValueValidator(5)])
    vehicleCategoryData1 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_cat_1')
    vehicleCategoryData2 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_cat_2')
    vehicleCategoryData3 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_cat_3')
    vehicleCategoryData4 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_cat_4')
    vehicleCategoryData5 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_cat_5')
    vehicle_category_data = models.JSONField(default=list)

    # Analysis Parameters
    resolution = models.IntegerField(default=0, validators=[MinValueValidator(
        1), MaxValueValidator(100)], blank=True, null=True)
    BR_F = models.CharField(max_length=10, default="", blank=True)
    shared_saving = models.IntegerField(default=0, blank=True, null=True)
    sum_pk_cost = models.FloatField(default=0, blank=True, null=True)
    sum_zero_cost = models.FloatField(default=0, blank=True, null=True)
    sum_op_cost = models.FloatField(default=0, blank=True, null=True)
    win_pk_cost = models.FloatField(default=0, blank=True, null=True)
    win_zero_cost = models.FloatField(default=0, blank=True, null=True)
    win_op_cost = models.FloatField(default=0, blank=True, null=True)
    summer_date = models.JSONField(default=list)
    winter_date = models.JSONField(default=list)
    s_pks = models.CharField(default="", blank=False,
                             null=False, max_length=10)
    s_pke = models.CharField(default="", blank=False,
                             null=False, max_length=10)
    s_sx = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    s_ops = models.CharField(default="", blank=False,
                             null=False, max_length=10)
    s_ope = models.CharField(default="", blank=False,
                             null=False, max_length=10)
    s_rb = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    w_pks = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_pke = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_sx = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    w_ops = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_ope = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_rb = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    date1_start = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date1_end = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date2_start = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date2_end = models.CharField(
        default="", blank=False, null=False, max_length=10)

    # Additional Fields from evAnalysis
    fileId = models.IntegerField(default=0, blank=False, null=False)
    user_name = models.CharField(
        default="", blank=False, null=False, max_length=50)

    def __str__(self):
        return f"Analysis {self.name} for {self.user.username} (ID: {self.id})"


class PermanentAnalysis(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='permanent_analyses'
    )
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    # Load Category Fields
    loadCategory = models.IntegerField(
        default=0, validators=[MinValueValidator(1), MaxValueValidator(6)])
    isLoadSplit = models.CharField(max_length=20, choices=(
        ("yes", 'Yes'), ("no", 'No')), default="")
    isLoadSplitFile = models.CharField(
        default="", blank=True, null=True, max_length=400)
    loadCategory1 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_load_cat_1')
    loadCategory2 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_load_cat_2')
    loadCategory3 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_load_cat_3')
    loadCategory4 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_load_cat_4')
    loadCategory5 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_load_cat_5')
    loadCategory6 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_load_cat_6')
    category_data = models.JSONField(default=list)

    # Vehicle Category Fields
    numOfvehicleCategory = models.IntegerField(blank=True, null=True, default=0, validators=[
                                               MinValueValidator(1), MaxValueValidator(5)])
    vehicleCategoryData1 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_vehicle_cat_1')
    vehicleCategoryData2 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_vehicle_cat_2')
    vehicleCategoryData3 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_vehicle_cat_3')
    vehicleCategoryData4 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_vehicle_cat_4')
    vehicleCategoryData5 = models.ForeignKey(
        VehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='permanent_vehicle_cat_5')
    vehicle_category_data = models.JSONField(default=list)


    # Analysis Parameters
    resolution = models.IntegerField(default=0, validators=[MinValueValidator(
        1), MaxValueValidator(100)], blank=True, null=True)
    BR_F = models.CharField(max_length=10, default="", blank=True)
    shared_saving = models.IntegerField(default=0, blank=True, null=True)
    sum_pk_cost = models.FloatField(default=0, blank=True, null=True)
    sum_zero_cost = models.FloatField(default=0, blank=True, null=True)
    sum_op_cost = models.FloatField(default=0, blank=True, null=True)
    win_pk_cost = models.FloatField(default=0, blank=True, null=True)
    win_zero_cost = models.FloatField(default=0, blank=True, null=True)
    win_op_cost = models.FloatField(default=0, blank=True, null=True)
    summer_date = models.JSONField(default=list)
    winter_date = models.JSONField(default=list)
    s_pks = models.CharField(default="", blank=False,
                             null=False, max_length=10)
    s_pke = models.CharField(default="", blank=False,
                             null=False, max_length=10)
    s_sx = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    s_ops = models.CharField(default="", blank=False,
                             null=False, max_length=10)
    s_ope = models.CharField(default="", blank=False,
                             null=False, max_length=10)
    s_rb = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    w_pks = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_pke = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_sx = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    w_ops = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_ope = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_rb = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    date1_start = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date1_end = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date2_start = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date2_end = models.CharField(
        default="", blank=False, null=False, max_length=10)
    # Paste the rest of the fields from Analysis except `fileId`

    user_name = models.CharField(
        default="", blank=True, null=True, max_length=50)

    def __str__(self):
            return f"Analysis {self.name} for {self.user.username} (ID: {self.id})"



class Files(models.Model):
    file = models.FileField(upload_to="FileUpload/", blank=False, null=False)

    def delete(self, *args, **kwargs):
        storage, path = self.file.storage, self.file.path
        super().delete(*args, **kwargs)
        storage.delete(path)

    def __str__(self):
        return f"File {self.id}"


class UserAnalysis(models.Model):
    userName = models.CharField(
        default="", blank=False, null=False, max_length=50)
    status = models.CharField(default="", blank=True, null=True, max_length=20)
    errorLog = models.TextField()
    time = models.FloatField(default=0.0, blank=True, null=True)

    class Meta:
        verbose_name_plural = "User Analyses Log"

    def __str__(self):
        return f"Analysis Log for {self.userName}"

