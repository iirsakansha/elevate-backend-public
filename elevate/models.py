from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
from elevate.constants import PERCENTAGE_VALIDATOR, CATEGORY_CHOICES, VEHICLE_CATEGORY_CHOICES
from django.core.exceptions import ValidationError
# Abstract base class for timestamped models


class TimeStampedModel(models.Model):
    """Abstract base class for models with created_at and updated_at fields."""
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class UserProfile(TimeStampedModel):
    """User profile model extending Django's User model."""
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="profile",
        verbose_name="user",
    )
    organization = models.CharField(
        max_length=255,
        default="Not assigned",
        verbose_name="organization",
    )
    invitation_status = models.CharField(
        max_length=50,
        default="Pending",
        verbose_name="invitation status",
    )
    invitation_username = models.CharField(
        max_length=255,
        blank=True,
        verbose_name="invitation username",
    )
    temporary_password = models.CharField(
        max_length=255,
        blank=True,
        verbose_name="temporary password",
    )
    invitation_token = models.CharField(
        max_length=255,
        unique=True,
        blank=True,
        verbose_name="invitation token",
    )
    password_reset_token = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        verbose_name="password reset token",
    )
    role = models.CharField(
        max_length=50,
        default="user",
        verbose_name="role",
    )

    class Meta:
        verbose_name = "user profile"
        verbose_name_plural = "user profiles"

    def __str__(self):
        return f"Profile for {self.user.username}"


class LoadCategoryModel(models.Model):
    """Model for categorizing load types."""
    category = models.CharField(
        max_length=20,
        choices=CATEGORY_CHOICES,
        default="",
        verbose_name="category",
    )
    category_file = models.CharField(
        max_length=400,
        default="",
        blank=True,
        null=True,
        verbose_name="category file",
    )
    sales_cagr = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="sales CAGR",
    )
    specify_split = models.FloatField(
        default=0,
        null=True,
        blank=True,
        verbose_name="specify split",
    )

    class Meta:
        verbose_name = "load category"
        verbose_name_plural = "load categories"

    def __str__(self):
        return self.category


class VehicleCategoryModel(models.Model):
    """Model for categorizing vehicle types."""
    vehicle_category = models.CharField(
        max_length=20,
        choices=VEHICLE_CATEGORY_CHOICES,
        default="",
        verbose_name="vehicle category",
    )
    vehicle_count = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="vehicle count",
    )
    fuel_efficiency = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="fuel efficiency",
    )
    cost_per_unit = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="cost per unit",
    )
    penetration_rate = models.FloatField(
        default=0,
        validators=PERCENTAGE_VALIDATOR,
        blank=True,
        verbose_name="penetration rate",
    )
    energy_consumption = models.FloatField(
        default=0.0,
        blank=True,
        null=True,
        verbose_name="energy consumption",
    )
    range_km = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="range (km)",
    )
    kwh_capacity = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="kWh capacity",
    )
    lifespan_years = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="lifespan (years)",
    )
    growth_rate = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="growth rate",
    )
    handling_cost = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="handling cost",
    )
    subsidy_amount = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="subsidy amount",
    )
    usage_factor = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="usage factor",
    )
    row_limit_xl = models.PositiveIntegerField(
        default=2_000_000,
        verbose_name="row limit (Excel)",
    )
    cagr_v = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="CAGR (vehicles)",
    )
    base_electricity_tariff = models.FloatField(
        default=0,
        validators=PERCENTAGE_VALIDATOR,
        blank=True,
        verbose_name="base electricity tariff",
    )

    class Meta:
        verbose_name = "vehicle category"
        verbose_name_plural = "vehicle categories"

    def __str__(self):
        return self.vehicle_category


class BaseAnalysisModel(TimeStampedModel):
    """Abstract base class for analysis models."""
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="%(class)s_analyses",
        verbose_name="user",
    )
    name = models.CharField(
        max_length=255,
        verbose_name="analysis name",
    )
    load_category_count = models.PositiveSmallIntegerField(
        default=0,
        validators=[MinValueValidator(1), MaxValueValidator(6)],
        verbose_name="load category count",
    )
    is_load_split = models.CharField(
        max_length=20,
        choices=(("yes", "Yes"), ("no", "No")),
        default="",
        verbose_name="is load split",
    )
    load_split_file = models.CharField(
        max_length=400,
        default="",
        blank=True,
        null=True,
        verbose_name="load split file",
    )
    load_categories = models.ManyToManyField(
        LoadCategoryModel,
        related_name="%(class)s_load_categories",
        blank=True,
        verbose_name="load categories",
    )
    category_data = models.JSONField(
        default=list,
        verbose_name="category data",
    )
    vehicle_category_count = models.PositiveSmallIntegerField(
        default=0,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        blank=True,
        null=True,
        verbose_name="vehicle category count",
    )
    vehicle_categories = models.ManyToManyField(
        VehicleCategoryModel,
        related_name="%(class)s_vehicle_categories",
        blank=True,
        verbose_name="vehicle categories",
    )
    vehicle_category_data = models.JSONField(
        default=list,
        verbose_name="vehicle category data",
    )
    resolution = models.PositiveSmallIntegerField(
        default=0,
        validators=[MinValueValidator(1), MaxValueValidator(100)],
        blank=True,
        null=True,
        verbose_name="resolution",
    )
    br_f = models.CharField(
        max_length=10,
        default="",
        blank=True,
        verbose_name="BR F",
    )
    shared_saving = models.IntegerField(
        default=0,
        blank=True,
        null=True,
        verbose_name="shared saving",
    )
    summer_peak_cost = models.FloatField(
        default=0,
        blank=True,
        null=True,
        verbose_name="summer peak cost",
    )
    summer_zero_cost = models.FloatField(
        default=0,
        blank=True,
        null=True,
        verbose_name="summer zero cost",
    )
    summer_op_cost = models.FloatField(
        default=0,
        blank=True,
        null=True,
        verbose_name="summer operating cost",
    )
    winter_peak_cost = models.FloatField(
        default=0,
        blank=True,
        null=True,
        verbose_name="winter peak cost",
    )
    winter_zero_cost = models.FloatField(
        default=0,
        blank=True,
        null=True,
        verbose_name="winter zero cost",
    )
    winter_op_cost = models.FloatField(
        default=0,
        blank=True,
        null=True,
        verbose_name="winter operating cost",
    )
    summer_date = models.JSONField(
        default=list,
        verbose_name="summer date",
    )
    winter_date = models.JSONField(
        default=list,
        verbose_name="winter date",
    )
    summer_peak_start = models.CharField(
        max_length=10,
        default="",
        blank=False,
        null=False,
        verbose_name="summer peak start",
    )
    summer_peak_end = models.CharField(
        max_length=10,
        default="",
        blank=False,
        null=False,
        verbose_name="summer peak end",
    )
    summer_sx = models.FloatField(
        default=0,
        validators=PERCENTAGE_VALIDATOR,
        blank=True,
        verbose_name="summer sx",
    )
    summer_op_start = models.CharField(
        max_length=10,
        default="",
        blank=False,
        null=False,
        verbose_name="summer operating start",
    )
    summer_op_end = models.CharField(
        max_length=10,
        default="",
        blank=False,
        null=False,
        verbose_name="summer operating end",
    )
    summer_rb = models.FloatField(
        default=0,
        validators=PERCENTAGE_VALIDATOR,
        blank=True,
        verbose_name="summer rb",
    )
    winter_peak_start = models.CharField(
        max_length=10,
        default="00:00",
        blank=False,
        null=False,
        verbose_name="winter peak start",
    )
    winter_peak_end = models.CharField(
        max_length=10,
        default="00:00",
        blank=False,
        null=False,
        verbose_name="winter peak end",
    )
    winter_sx = models.FloatField(
        default=0,
        validators=PERCENTAGE_VALIDATOR,
        blank=True,
        verbose_name="winter sx",
    )
    winter_op_start = models.CharField(
        max_length=10,
        default="00:00",
        blank=False,
        null=False,
        verbose_name="winter operating start",
    )
    winter_op_end = models.CharField(
        max_length=10,
        default="00:00",
        blank=False,
        null=False,
        verbose_name="winter operating end",
    )
    winter_rb = models.FloatField(
        default=0,
        validators=PERCENTAGE_VALIDATOR,
        blank=True,
        verbose_name="winter rb",
    )
    date1_start = models.CharField(
        max_length=10,
        default="",
        blank=False,
        null=False,
        verbose_name="date 1 start",
    )
    date1_end = models.CharField(
        max_length=10,
        default="",
        blank=False,
        null=False,
        verbose_name="date 1 end",
    )
    date2_start = models.CharField(
        max_length=10,
        default="",
        blank=False,
        null=False,
        verbose_name="date 2 start",
    )
    date2_end = models.CharField(
        max_length=10,
        default="",
        blank=False,
        null=False,
        verbose_name="date 2 end",
    )
    user_name = models.CharField(
        max_length=50,
        default="",
        blank=True,
        null=True,
        verbose_name="user name",
    )

    def clean(self):
        if not isinstance(self.category_data, list):
            raise ValidationError("category_data must be a list")
        if not isinstance(self.vehicle_category_data, list):
            raise ValidationError("vehicle_category_data must be a list")

    class Meta:
        abstract = True


class Analysis(BaseAnalysisModel):
    """Model for user-specific analyses."""
    file_id = models.PositiveIntegerField(
        default=0,
        verbose_name="file ID",
    )

    class Meta:
        verbose_name = "analysis"
        verbose_name_plural = "analyses"

    def __str__(self):
        return f"Analysis {self.name} for {self.user.username} (ID: {self.id})"


class PermanentAnalysis(BaseAnalysisModel):
    """Model for permanent analyses with nullable user."""
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_analyses",
        verbose_name="user",
    )

    class Meta:
        verbose_name = "permanent analysis"
        verbose_name_plural = "permanent analyses"

    def __str__(self):
        return f"Permanent Analysis {self.name} for {self.user.username if self.user else 'Anonymous'} (ID: {self.id})"


class Files(models.Model):
    """Model for storing uploaded files."""
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="uploaded_files",
        verbose_name="user",
    )
    file = models.FileField(
        upload_to="file_upload/",
        blank=False,
        null=False,
        verbose_name="file",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="created at",
    )

    class Meta:
        verbose_name = "file"
        verbose_name_plural = "files"

    def delete(self, *args, **kwargs):
        """Delete the file from storage and database."""
        try:
            storage, path = self.file.storage, self.file.path
            super().delete(*args, **kwargs)
            storage.delete(path)
        except Exception as e:
            # Log the error (assuming a logging setup)
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error deleting file {self.file.name}: {str(e)}")

    def __str__(self):
        return f"File {self.id} uploaded by {self.user.username}"

class UserAnalysis(models.Model):
    """Model for logging user analysis activities."""
    user_name = models.CharField(
        max_length=50,
        default="",
        blank=False,
        null=False,
        verbose_name="user name",
    )
    status = models.CharField(
        max_length=20,
        default="",
        blank=True,
        null=True,
        verbose_name="status",
    )
    error_log = models.TextField(
        verbose_name="error log",
    )
    time = models.FloatField(
        default=0.0,
        blank=True,
        null=True,
        verbose_name="time",
    )

    class Meta:
        verbose_name = "user analysis log"
        verbose_name_plural = "user analyses log"

    def __str__(self):
        return f"Analysis Log for {self.user_name}"
