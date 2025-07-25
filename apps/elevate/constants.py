# myapp/constants.py
from django.core.validators import MaxValueValidator, MinValueValidator

PERCENTAGE_VALIDATOR = [MinValueValidator(0), MaxValueValidator(100)]
CATEGORY_CHOICES = (
    ("commercial", "Commercial"),
    ("agricultural", "Agricultural"),
    ("industrial", "Industrial"),
    ("residential", "Residential"),
    ("public", "Public"),
    ("others", "Others"),
)
VEHICLE_CATEGORY_CHOICES = (
    ("car", "Car"),
    ("bus", "Bus"),
    ("2-wheeler", "2-Wheeler"),
    ("3-wheeler", "3-Wheeler"),
    ("others", "Others"),
)
