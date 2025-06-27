# myapp/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class UserProfile(models.Model):
    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name='profile')
    organization = models.CharField(max_length=255, default='Not assigned')
    invitation_status = models.CharField(max_length=50, default='Pending')
    invitation_username = models.CharField(max_length=255, blank=True)
    temporary_password = models.CharField(max_length=255, blank=True)
    invitation_token = models.CharField(
        max_length=255, unique=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Profile for {self.user.username}"
    

class Template(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='templates')
    name = models.CharField(max_length=255)
    dateCreated = models.DateTimeField(default=timezone.now)
    load_category = models.IntegerField()
    is_load_split = models.CharField(max_length=10)
    is_load_split_file = models.CharField(max_length=255, blank=True)
    category_data = models.JSONField()
    num_of_vehicle_category = models.IntegerField()
    vehicle_category_data = models.JSONField()
    resolution = models.IntegerField()
    br_f = models.CharField(max_length=10)
    shared_saving = models.IntegerField()
    sum_pk_cost = models.FloatField()
    sum_zero_cost = models.FloatField()
    sum_op_cost = models.FloatField()
    win_pk_cost = models.FloatField()
    win_zero_cost = models.FloatField()
    win_op_cost = models.FloatField()
    summer_date = models.JSONField()
    winter_date = models.JSONField()
    s_pks = models.CharField(max_length=10)
    s_pke = models.CharField(max_length=10)
    s_sx = models.CharField(max_length=10)
    s_ops = models.CharField(max_length=10)
    s_ope = models.CharField(max_length=10)
    s_rb = models.CharField(max_length=10)
    w_pks = models.CharField(max_length=10)
    w_pke = models.CharField(max_length=10)
    w_sx = models.CharField(max_length=10)
    w_ops = models.CharField(max_length=10)
    w_ope = models.CharField(max_length=10)
    w_rb = models.CharField(max_length=10)
    date1_start = models.CharField(max_length=10)
    date1_end = models.CharField(max_length=10)
    date2_start = models.CharField(max_length=10)
    date2_end = models.CharField(max_length=10)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Template {self.name} for {self.user.username} (ID: {self.id})"
