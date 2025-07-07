from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .models import (
    PermanentAnalysis, UserProfile, LoadCategoryModel,
    VehicleCategoryModel, Analysis, Files, UserAnalysis
)
from datetime import datetime


class RegisterSerializer(serializers.ModelSerializer):
    """Serializer for user registration."""
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        """Create a new user and associated profile."""
        user = User.objects.create_user(
            validated_data['username'],
            validated_data['email'],
            validated_data['password']
        )
        UserProfile.objects.create(user=user)
        return user


class LoginUserSerializer(serializers.Serializer):
    """Serializer for user login."""
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        """Validate user credentials."""
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError("Invalid credentials")


class ChangePasswordSerializer(serializers.Serializer):
    """Serializer for changing user password."""
    old_password = serializers.CharField(required=True, write_only=True)
    new_password = serializers.CharField(
        required=True, write_only=True, min_length=6)


class SetPasswordSerializer(serializers.Serializer):
    """Serializer for setting a new password for invited users."""
    new_password = serializers.CharField(min_length=6, write_only=True)


class PasswordResetSerializer(serializers.Serializer):
    """Serializer for password reset request."""
    email = serializers.EmailField()


class PasswordResetNoEmailSerializer(serializers.Serializer):
    """Serializer for password reset with token."""
    new_password = serializers.CharField(min_length=6, write_only=True)
    uidb64 = serializers.CharField()
    token = serializers.CharField()


class InvitedUserProfileSerializer(serializers.ModelSerializer):
    """Serializer for invited user profile."""
    organization = serializers.CharField(
        source='profile.organization', default='Not assigned')
    status = serializers.CharField(
        source='profile.invitation_status', default='Pending')
    created_at = serializers.DateTimeField(
        source='profile.created_at', read_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'email',
                  'organization', 'status', 'created_at']

    def validate(self, data):
        """Validate that the user account is active."""
        user = self.instance
        if user and not user.is_active:
            raise serializers.ValidationError("Account not activated")
        return data


class UserSerializer(serializers.ModelSerializer):
    """Serializer for user details."""
    class Meta:
        model = User
        fields = ['id', 'last_login', 'first_name',
                  'last_name', 'username', 'email']


class LoadCategoryModelSerializer(serializers.ModelSerializer):
    """Serializer for LoadCategoryModel."""
    class Meta:
        model = LoadCategoryModel
        fields = ['id', 'category', 'category_file',
                  'sales_cagr', 'specify_split']

    def validate(self, data):
        """Validate load category data."""
        if data.get('sales_cagr', 0) < 0:
            raise serializers.ValidationError(
                "sales_cagr must be non-negative")
        if data.get('specify_split', 0) < 0:
            raise serializers.ValidationError(
                "specify_split must be non-negative")
        return data


class VehicleCategoryModelSerializer(serializers.ModelSerializer):
    """Serializer for VehicleCategoryModel."""
    class Meta:
        model = VehicleCategoryModel
        fields = [
            'id', 'vehicle_category', 'vehicle_count', 'fuel_efficiency', 'cost_per_unit',
            'penetration_rate', 'energy_consumption', 'range_km', 'kwh_capacity',
            'lifespan_years', 'growth_rate', 'handling_cost', 'subsidy_amount',
            'usage_factor', 'row_limit_xl', 'cagr_v', 'base_electricity_tariff'
        ]

    def validate(self, data):
        """Validate vehicle category data."""
        numeric_fields = [
            'vehicle_count', 'fuel_efficiency', 'cost_per_unit', 'penetration_rate',
            'energy_consumption', 'range_km', 'kwh_capacity', 'lifespan_years',
            'growth_rate', 'handling_cost', 'subsidy_amount', 'usage_factor',
            'cagr_v', 'base_electricity_tariff'
        ]
        for field in numeric_fields:
            if field in data and data[field] is not None and data[field] < 0:
                raise serializers.ValidationError(
                    f"{field} must be non-negative")
        if 'fuel_efficiency' in data and data['fuel_efficiency'] > 100:
            raise serializers.ValidationError(
                "fuel_efficiency must be between 0 and 100")
        if 'subsidy_amount' in data and data['subsidy_amount'] > 100:
            raise serializers.ValidationError(
                "subsidy_amount must be between 0 and 100")
        if 'handling_cost' in data and data['handling_cost'] > 100:
            raise serializers.ValidationError(
                "handling_cost must be between 0 and 100")
        if 'cagr_v' in data and data['cagr_v'] > 100:
            raise serializers.ValidationError(
                "cagr_v must be between 0 and 100")
        return data


class PermanentAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for PermanentAnalysis."""
    category_data = LoadCategoryModelSerializer(many=True)
    vehicle_category_data = VehicleCategoryModelSerializer(many=True)
    user = UserSerializer(read_only=True)

    class Meta:
        model = PermanentAnalysis
        fields = [
            'id', 'user', 'name', 'load_category_count', 'load_split_file',
            'category_data', 'vehicle_category_count', 'vehicle_category_data',
            'resolution', 'br_f', 'shared_saving', 'summer_peak_cost',
            'summer_zero_cost', 'summer_op_cost', 'winter_peak_cost',
            'winter_zero_cost', 'winter_op_cost', 'summer_date', 'winter_date',
            'summer_peak_start', 'summer_peak_end', 'summer_op_start',
            'summer_op_end', 'winter_peak_start', 'winter_peak_end',
            'winter_op_start', 'winter_op_end', 'summer_sx', 'summer_rb',
            'winter_sx', 'winter_rb', 'date1_start', 'date1_end', 'date2_start',
            'date2_end', 'created_at', 'updated_at'
        ]

    def validate(self, data):
        """Validate PermanentAnalysis data."""
        # Validate load_category_count and category_data
        load_category_count = data.get('load_category_count', 0)
        category_data = data.get('category_data', [])
        if load_category_count != len(category_data):
            raise serializers.ValidationError(
                "load_category_count must match the number of category_data entries"
            )
        if load_category_count < 1 or load_category_count > 6:
            raise serializers.ValidationError(
                "load_category_count must be between 1 and 6"
            )
        total_split = sum(item.get('specify_split', 0)
                          for item in category_data)
        if abs(total_split - 100) > 0.01:
            raise serializers.ValidationError(
                "Sum of specify_split in category_data must be 100%"
            )

        # Validate vehicle_category_count and vehicle_category_data
        vehicle_category_count = data.get('vehicle_category_count', 0)
        vehicle_category_data = data.get('vehicle_category_data', [])
        if vehicle_category_count != len(vehicle_category_data):
            raise serializers.ValidationError(
                "vehicle_category_count must match the number of vehicle_category_data entries"
            )
        if vehicle_category_count < 1 or vehicle_category_count > 5:
            raise serializers.ValidationError(
                "vehicle_category_count must be between 1 and 5"
            )

        # Validate numeric fields
        numeric_fields = [
            'resolution', 'shared_saving', 'summer_peak_cost', 'summer_zero_cost',
            'summer_op_cost', 'winter_peak_cost', 'winter_zero_cost', 'winter_op_cost',
            'summer_sx', 'summer_rb', 'winter_sx', 'winter_rb'
        ]
        for field in numeric_fields:
            if field in data and data[field] is not None and data[field] < 0:
                raise serializers.ValidationError(
                    f"{field} must be non-negative")
        if 'shared_saving' in data and data['shared_saving'] > 100:
            raise serializers.ValidationError(
                "shared_saving must be between 0 and 100")
        if 'summer_sx' in data and data['summer_sx'] > 100:
            raise serializers.ValidationError(
                "summer_sx must be between 0 and 100")
        if 'summer_rb' in data and data['summer_rb'] > 100:
            raise serializers.ValidationError(
                "summer_rb must be between 0 and 100")
        if 'winter_sx' in data and data['winter_sx'] > 100:
            raise serializers.ValidationError(
                "winter_sx must be between 0 and 100")
        if 'winter_rb' in data and data['winter_rb'] > 100:
            raise serializers.ValidationError(
                "winter_rb must be between 0 and 100")

        # Validate date fields
        for date_field in ['summer_date', 'winter_date']:
            if date_field in data and not isinstance(data[date_field], list):
                raise serializers.ValidationError(
                    f"{date_field} must be a list")
        for date_field in ['date1_start', 'date1_end', 'date2_start', 'date2_end']:
            if date_field in data and data[date_field]:
                try:
                    datetime.strptime(data[date_field], '%b-%d')
                except ValueError:
                    raise serializers.ValidationError(
                        f"{date_field} must be in MMM-DD format")

        # Validate time fields
        time_fields = [
            'summer_peak_start', 'summer_peak_end', 'summer_op_start', 'summer_op_end',
            'winter_peak_start', 'winter_peak_end', 'winter_op_start', 'winter_op_end'
        ]
        for time_field in time_fields:
            if time_field in data and data[time_field]:
                try:
                    datetime.strptime(data[time_field], '%H:%M')
                except ValueError:
                    raise serializers.ValidationError(
                        f"{time_field} must be in HH:MM format")

        return data

    def create(self, validated_data):
        """Create a new PermanentAnalysis instance."""
        category_data = validated_data.pop('category_data', [])
        vehicle_category_data = validated_data.pop('vehicle_category_data', [])
        instance = PermanentAnalysis.objects.create(**validated_data)
        for item in category_data:
            LoadCategoryModel.objects.create(analysis=instance, **item)
        for item in vehicle_category_data:
            VehicleCategoryModel.objects.create(analysis=instance, **item)
        return instance

    def update(self, instance, validated_data):
        """Update an existing PermanentAnalysis instance."""
        category_data = validated_data.pop('category_data', [])
        vehicle_category_data = validated_data.pop('vehicle_category_data', [])
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        instance.category_data.all().delete()
        instance.vehicle_category_data.all().delete()
        for item in category_data:
            LoadCategoryModel.objects.create(analysis=instance, **item)
        for item in vehicle_category_data:
            VehicleCategoryModel.objects.create(analysis=instance, **item)
        return instance


class AnalysisSerializer(serializers.ModelSerializer):
    """Serializer for Analysis."""
    category_data = LoadCategoryModelSerializer(many=True)
    vehicle_category_data = VehicleCategoryModelSerializer(many=True)
    user = UserSerializer(read_only=True)

    class Meta:
        model = Analysis
        fields = [
            'id', 'user', 'name', 'load_category_count', 'load_split_file',
            'category_data', 'vehicle_category_count', 'vehicle_category_data',
            'resolution', 'br_f', 'shared_saving', 'summer_peak_cost',
            'summer_zero_cost', 'summer_op_cost', 'winter_peak_cost',
            'winter_zero_cost', 'winter_op_cost', 'summer_date', 'winter_date',
            'summer_peak_start', 'summer_peak_end', 'summer_op_start',
            'summer_op_end', 'winter_peak_start', 'winter_peak_end',
            'winter_op_start', 'winter_op_end', 'summer_sx', 'summer_rb',
            'winter_sx', 'winter_rb', 'date1_start', 'date1_end', 'date2_start',
            'date2_end', 'created_at', 'updated_at'
        ]

    def validate(self, data):
        """Validate Analysis data."""
        # Same validation as PermanentAnalysisSerializer
        load_category_count = data.get('load_category_count', 0)
        category_data = data.get('category_data', [])
        if load_category_count != len(category_data):
            raise serializers.ValidationError(
                "load_category_count must match the number of category_data entries"
            )
        if load_category_count < 1 or load_category_count > 6:
            raise serializers.ValidationError(
                "load_category_count must be between 1 and 6"
            )
        total_split = sum(item.get('specify_split', 0)
                          for item in category_data)
        if abs(total_split - 100) > 0.01:
            raise serializers.ValidationError(
                "Sum of specify_split in category_data must be 100%"
            )

        vehicle_category_count = data.get('vehicle_category_count', 0)
        vehicle_category_data = data.get('vehicle_category_data', [])
        if vehicle_category_count != len(vehicle_category_data):
            raise serializers.ValidationError(
                "vehicle_category_count must match the number of vehicle_category_data entries"
            )
        if vehicle_category_count < 1 or vehicle_category_count > 5:
            raise serializers.ValidationError(
                "vehicle_category_count must be between 1 and 5"
            )

        numeric_fields = [
            'resolution', 'shared_saving', 'summer_peak_cost', 'summer_zero_cost',
            'summer_op_cost', 'winter_peak_cost', 'winter_zero_cost', 'winter_op_cost',
            'summer_sx', 'summer_rb', 'winter_sx', 'winter_rb'
        ]
        for field in numeric_fields:
            if field in data and data[field] is not None and data[field] < 0:
                raise serializers.ValidationError(
                    f"{field} must be non-negative")
        if 'shared_saving' in data and data['shared_saving'] > 100:
            raise serializers.ValidationError(
                "shared_saving must be between 0 and 100")
        if 'summer_sx' in data and data['summer_sx'] > 100:
            raise serializers.ValidationError(
                "summer_sx must be between 0 and 100")
        if 'summer_rb' in data and data['summer_rb'] > 100:
            raise serializers.ValidationError(
                "summer_rb must be between 0 and 100")
        if 'winter_sx' in data and data['winter_sx'] > 100:
            raise serializers.ValidationError(
                "winter_sx must be between 0 and 100")
        if 'winter_rb' in data and data['winter_rb'] > 100:
            raise serializers.ValidationError(
                "winter_rb must be between 0 and 100")

        for date_field in ['summer_date', 'winter_date']:
            if date_field in data and not isinstance(data[date_field], list):
                raise serializers.ValidationError(
                    f"{date_field} must be a list")
        for date_field in ['date1_start', 'date1_end', 'date2_start', 'date2_end']:
            if date_field in data and data[date_field]:
                try:
                    datetime.strptime(data[date_field], '%b-%d')
                except ValueError:
                    raise serializers.ValidationError(
                        f"{date_field} must be in MMM-DD format")

        time_fields = [
            'summer_peak_start', 'summer_peak_end', 'summer_op_start', 'summer_op_end',
            'winter_peak_start', 'winter_peak_end', 'winter_op_start', 'winter_op_end'
        ]
        for time_field in time_fields:
            if time_field in data and data[time_field]:
                try:
                    datetime.strptime(data[time_field], '%H:%M')
                except ValueError:
                    raise serializers.ValidationError(
                        f"{time_field} must be in HH:MM format")

        return data

    def create(self, validated_data):
        """Create a new Analysis instance."""
        category_data = validated_data.pop('category_data', [])
        vehicle_category_data = validated_data.pop('vehicle_category_data', [])
        instance = Analysis.objects.create(**validated_data)
        for item in category_data:
            LoadCategoryModel.objects.create(analysis=instance, **item)
        for item in vehicle_category_data:
            VehicleCategoryModel.objects.create(analysis=instance, **item)
        return instance

    def update(self, instance, validated_data):
        """Update an existing Analysis instance."""
        category_data = validated_data.pop('category_data', [])
        vehicle_category_data = validated_data.pop('vehicle_category_data', [])
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        instance.category_data.all().delete()
        instance.vehicle_category_data.all().delete()
        for item in category_data:
            LoadCategoryModel.objects.create(analysis=instance, **item)
        for item in vehicle_category_data:
            VehicleCategoryModel.objects.create(analysis=instance, **item)
        return instance


class FilesSerializer(serializers.ModelSerializer):
    """Serializer for Files."""
    class Meta:
        model = Files
        fields = ['id', 'user', 'file', 'created_at']


class UserAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for UserAnalysis."""
    class Meta:
        model = UserAnalysis
        fields = ['id', 'user', 'analysis', 'created_at']
