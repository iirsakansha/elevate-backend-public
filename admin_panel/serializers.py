from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile


class SetPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField(required=False)  # Added for password reset
    old_password = serializers.CharField(
        write_only=True, required=False)  # Optional for reset
    new_password = serializers.CharField(write_only=True)

    def validate(self, data):
        user = self.context['request'].user if self.context.get(
            'request') else None
        email = data.get('email')
        old_password = data.get('old_password')
        new_password = data.get('new_password')

        # For password reset (no user authentication, email provided)
        if email and not user:
            if not new_password:
                raise serializers.ValidationError(
                    {"new_password": "New password is required"})
            if len(new_password) < 8:
                raise serializers.ValidationError(
                    {"new_password": "Password must be at least 8 characters long"})
            return data

        # For password change (authenticated user, old_password required)
        if not user or not user.is_authenticated:
            raise serializers.ValidationError(
                "User must be authenticated for password change")
        if not old_password:
            raise serializers.ValidationError(
                {"old_password": "Old password is required"})
        if not user.check_password(old_password):
            raise serializers.ValidationError(
                {"old_password": "Incorrect old password"})
        if len(new_password) < 8:
            raise serializers.ValidationError(
                {"new_password": "Password must be at least 8 characters long"})
        return data

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
