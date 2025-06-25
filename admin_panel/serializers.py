from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile


class SetPasswordSerializer(serializers.Serializer):
    new_password = serializers.CharField(min_length=6, write_only=True)


class PasswordResetSerializer(serializers.Serializer):
    email = serializers.EmailField()
    new_password = serializers.CharField(min_length=6, write_only=True)
   

class PasswordResetNoEmailSerializer(serializers.Serializer):
    new_password = serializers.CharField(min_length=6)
    uidb64 = serializers.CharField()
    token = serializers.CharField()

class InvitedUserProfileSerializer(serializers.ModelSerializer):
    organization = serializers.CharField(
        source='profile.organization', default='Not assigned')
    status = serializers.CharField(
        source='profile.invitation_status', default='Pending')
    created_at = serializers.DateTimeField(  # Add created_at field
        source='profile.created_at', read_only=True
    )

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'organization',
                  'status', 'created_at']  # Include created_at

    def validate(self, data):
        user = self.instance
        if user and not user.is_active:
            raise serializers.ValidationError("Account not activated")
        return data
