from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

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
