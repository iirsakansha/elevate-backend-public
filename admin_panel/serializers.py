# myapp/serializers.py
from .models import Template
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, Template
import re

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
    created_at = serializers.DateTimeField(
        source='profile.created_at', read_only=True
    )

    class Meta:
        model = User
        fields = ['id', 'username', 'email',
                  'organization', 'status', 'created_at']

    def validate(self, data):
        user = self.instance
        if user and not user.is_active:
            raise serializers.ValidationError("Account not activated")
        return data


class TemplateSerializer(serializers.ModelSerializer):
    formData = serializers.JSONField(write_only=True, required=False)

    class Meta:
        model = Template
        fields = [
            'id', 'name', 'dateCreated', 'formData',
            'load_category', 'is_load_split', 'is_load_split_file', 'category_data',
            'num_of_vehicle_category', 'vehicle_category_data',
            'resolution', 'br_f', 'shared_saving', 'sum_pk_cost', 'sum_zero_cost', 'sum_op_cost',
            'win_pk_cost', 'win_zero_cost', 'win_op_cost',
            'summer_date', 'winter_date', 's_pks', 's_pke', 's_sx', 's_ops', 's_ope', 's_rb',
            'w_pks', 'w_pke', 'w_sx', 'w_ops', 'w_ope', 'w_rb',
            'date1_start', 'date1_end', 'date2_start', 'date2_end'
        ]
        read_only_fields = ['id', 'dateCreated']
        extra_kwargs = {
            'load_category': {'required': False},
            'is_load_split': {'required': False},
            'is_load_split_file': {'required': False, 'allow_blank': True},
            'category_data': {'required': False},
            'num_of_vehicle_category': {'required': False},
            'vehicle_category_data': {'required': False},
            'resolution': {'required': False},
            'br_f': {'required': False, 'allow_blank': True},
            'shared_saving': {'required': False},
            'sum_pk_cost': {'required': False},
            'sum_zero_cost': {'required': False},
            'sum_op_cost': {'required': False},
            'win_pk_cost': {'required': False},
            'win_zero_cost': {'required': False},
            'win_op_cost': {'required': False},
            'summer_date': {'required': False},
            'winter_date': {'required': False},
            's_pks': {'required': False, 'allow_blank': True},
            's_pke': {'required': False, 'allow_blank': True},
            's_sx': {'required': False, 'allow_blank': True},
            's_ops': {'required': False, 'allow_blank': True},
            's_ope': {'required': False, 'allow_blank': True},
            's_rb': {'required': False, 'allow_blank': True},
            'w_pks': {'required': False, 'allow_blank': True},
            'w_pke': {'required': False, 'allow_blank': True},
            'w_sx': {'required': False, 'allow_blank': True},
            'w_ops': {'required': False, 'allow_blank': True},
            'w_ope': {'required': False, 'allow_blank': True},
            'w_rb': {'required': False, 'allow_blank': True},
            'date1_start': {'required': False, 'allow_blank': True},
            'date1_end': {'required': False, 'allow_blank': True},
            'date2_start': {'required': False, 'allow_blank': True},
            'date2_end': {'required': False, 'allow_blank': True},
        }

    def to_representation(self, instance):
        """Convert model instance to JSON representation"""
        data = super().to_representation(instance)
        data['formData'] = {
            'form1': {
                'load_category': instance.load_category,
                'is_load_split': instance.is_load_split,
                'is_load_split_file': instance.is_load_split_file,
                'category_data': instance.category_data,
            },
            'form2': {
                'num_of_vehicle_category': instance.num_of_vehicle_category,
                'vehicle_category_data': instance.vehicle_category_data,
            },
            'form3': {
                'resolution': instance.resolution,
                'br_f': instance.br_f,
                'shared_saving': instance.shared_saving,
                'sum_pk_cost': instance.sum_pk_cost,
                'sum_zero_cost': instance.sum_zero_cost,
                'sum_op_cost': instance.sum_op_cost,
                'win_pk_cost': instance.win_pk_cost,
                'win_zero_cost': instance.win_zero_cost,
                'win_op_cost': instance.win_op_cost,
            },
            'form4': {
                'summer_date': instance.summer_date,
                'winter_date': instance.winter_date,
                's_pks': instance.s_pks,
                's_pke': instance.s_pke,
                's_sx': instance.s_sx,
                's_ops': instance.s_ops,
                's_ope': instance.s_ope,
                's_rb': instance.s_rb,
                'w_pks': instance.w_pks,
                'w_pke': instance.w_pke,
                'w_sx': instance.w_sx,
                'w_ops': instance.w_ops,
                'w_ope': instance.w_ope,
                'w_rb': instance.w_rb,
                'date1_start': instance.date1_start,
                'date1_end': instance.date1_end,
                'date2_start': instance.date2_start,
                'date2_end': instance.date2_end,
            },
        }
        fields_to_remove = [
            'load_category', 'is_load_split', 'is_load_split_file', 'category_data',
            'num_of_vehicle_category', 'vehicle_category_data',
            'resolution', 'br_f', 'shared_saving', 'sum_pk_cost', 'sum_zero_cost', 'sum_op_cost',
            'win_pk_cost', 'win_zero_cost', 'win_op_cost',
            'summer_date', 'winter_date', 's_pks', 's_pke', 's_sx', 's_ops', 's_ope', 's_rb',
            'w_pks', 'w_pke', 'w_sx', 'w_ops', 'w_ope', 'w_rb',
            'date1_start', 'date1_end', 'date2_start', 'date2_end'
        ]
        for field in fields_to_remove:
            data.pop(field, None)
        return data

    def create(self, validated_data):
        form_data = validated_data.pop('formData', {})
        if form_data:
            self._extract_form_data(validated_data, form_data)
        self._set_defaults(validated_data)
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)

    def update(self, instance, validated_data):
        form_data = validated_data.pop('formData', {})
        if form_data:
            self._extract_form_data(validated_data, form_data)
        validated_data['user'] = instance.user
        return super().update(instance, validated_data)

    def _extract_form_data(self, validated_data, form_data):
        form1 = form_data.get('form1', {})
        validated_data['load_category'] = int(form1.get('load_category', 0))
        validated_data['is_load_split'] = form1.get('is_load_split', '')
        validated_data['is_load_split_file'] = form1.get(
            'is_load_split_file', '')
        validated_data['category_data'] = form1.get('category_data', [])
        form2 = form_data.get('form2', {})
        validated_data['num_of_vehicle_category'] = int(
            form2.get('num_of_vehicle_category', 0))
        validated_data['vehicle_category_data'] = form2.get(
            'vehicle_category_data', [])
        form3 = form_data.get('form3', {})
        validated_data['resolution'] = int(form3.get('resolution', 0))
        validated_data['br_f'] = form3.get('br_f', '')
        validated_data['shared_saving'] = int(form3.get('shared_saving', 0))
        for field in ['sum_pk_cost', 'sum_zero_cost', 'sum_op_cost', 'win_pk_cost', 'win_zero_cost', 'win_op_cost']:
            try:
                validated_data[field] = float(form3.get(field, 0.0) or 0.0)
            except (ValueError, TypeError):
                validated_data[field] = 0.0
        form4 = form_data.get('form4', {})
        validated_data['summer_date'] = form4.get('summer_date', [])
        validated_data['winter_date'] = form4.get('winter_date', [])
        validated_data['s_pks'] = form4.get('s_pks', '')
        validated_data['s_pke'] = form4.get('s_pke', '')
        validated_data['s_sx'] = form4.get('s_sx', '')
        validated_data['s_ops'] = form4.get('s_ops', '')
        validated_data['s_ope'] = form4.get('s_ope', '')
        validated_data['s_rb'] = form4.get('s_rb', '')
        validated_data['w_pks'] = form4.get('w_pks', '')
        validated_data['w_pke'] = form4.get('w_pke', '')
        validated_data['w_sx'] = form4.get('w_sx', '')
        validated_data['w_ops'] = form4.get('w_ops', '')
        validated_data['w_ope'] = form4.get('w_ope', '')
        validated_data['w_rb'] = form4.get('w_rb', '')
        validated_data['date1_start'] = form4.get('date1_start', '')[:10]
        validated_data['date1_end'] = form4.get('date1_end', '')[:10]
        validated_data['date2_start'] = form4.get('date2_start', '')[:10]
        validated_data['date2_end'] = form4.get('date2_end', '')[:10]

    def _set_defaults(self, validated_data):
        defaults = {
            'load_category': 0,
            'is_load_split': '',
            'is_load_split_file': '',
            'category_data': [],
            'num_of_vehicle_category': 0,
            'vehicle_category_data': [],
            'resolution': 0,
            'br_f': '',
            'shared_saving': 0,
            'sum_pk_cost': 0.0,
            'sum_zero_cost': 0.0,
            'sum_op_cost': 0.0,
            'win_pk_cost': 0.0,
            'win_zero_cost': 0.0,
            'win_op_cost': 0.0,
            'summer_date': [],
            'winter_date': [],
            's_pks': '',
            's_pke': '',
            's_sx': '',
            's_ops': '',
            's_ope': '',
            's_rb': '',
            'w_pks': '',
            'w_pke': '',
            'w_sx': '',
            'w_ops': '',
            'w_ope': '',
            'w_rb': '',
            'date1_start': '',
            'date1_end': '',
            'date2_start': '',
            'date2_end': '',
        }
        for field, default_value in defaults.items():
            if field not in validated_data:
                validated_data[field] = default_value

    def validate_category_data(self, value):
        if not isinstance(value, list):
            raise serializers.ValidationError("category_data must be a list")
        for item in value:
            if not all(k in item for k in ['category', 'specifySplit', 'salesCAGR']):
                raise serializers.ValidationError(
                    "Each category_data item must contain 'category', 'specifySplit', and 'salesCAGR'")
        return value

    def validate_vehicle_category_data(self, value):
        if not isinstance(value, list):
            raise serializers.ValidationError(
                "vehicle_category_data must be a list")
        required_keys = ['vehicleCategory', 'n', 'f', 'c', 'p', 'e', 'r',
                         'k', 'l', 'g', 'h', 's', 'u', 'CAGR_V', 'baseElectricityTariff']
        for item in value:
            if not all(k in item for k in required_keys):
                raise serializers.ValidationError(
                    f"Each vehicle_category_data item must contain all required keys: {required_keys}")
        return value

    def validate_summer_date(self, value):
        if not isinstance(value, list) or len(value) != 2:
            raise serializers.ValidationError(
                "summer_date must be a list with exactly 2 dates")
        return value

    def validate_winter_date(self, value):
        if not isinstance(value, list) or len(value) != 2:
            raise serializers.ValidationError(
                "winter_date must be a list with exactly 2 dates")
        return value

    def _validate_time_format(self, value, field_name):
        if value and not re.match(r"^\d{2}:\d{2}$", value):
            raise serializers.ValidationError(
                f"{field_name} must be in HH:mm format")
        return value

    def validate_s_pks(self, value):
        return self._validate_time_format(value, "s_pks")

    def validate_s_pke(self, value):
        return self._validate_time_format(value, "s_pke")

    def validate_s_ops(self, value):
        return self._validate_time_format(value, "s_ops")

    def validate_s_ope(self, value):
        return self._validate_time_format(value, "s_ope")

    def validate_w_pks(self, value):
        return self._validate_time_format(value, "w_pks")

    def validate_w_pke(self, value):
        return self._validate_time_format(value, "w_pke")

    def validate_w_ops(self, value):
        return self._validate_time_format(value, "w_ops")

    def validate_w_ope(self, value):
        return self._validate_time_format(value, "w_ope")
