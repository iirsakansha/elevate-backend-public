from django.http import response
from rest_framework import serializers
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import evAnalysis, LoadCategoryModel, vehicleCategoryModel, Files
from django.contrib.auth import authenticate
from rest_framework.response import Response

# Register Serializer


class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password')
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(
            validated_data['username'], validated_data['email'], validated_data['password'])

        return user

# Change Paswword


class ChangePasswordSerializer(serializers.Serializer):
    model = User

    """
    Serializer for password change endpoint.
    """
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)

# Change Profile


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'last_login', 'first_name',
                  'last_name', 'username', 'email')


class LoginUserSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

    def validate(self, data):
        user = authenticate(**data)
        if user and user.is_active:
            return user
        raise serializers.ValidationError(
            "The user_id or password you entered is wrong. You may need to contact admin")


class LoadCategoryModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = LoadCategoryModel
        fields = ['id', 'category', 'categoryFile',
                  'salesCAGR', 'specifySplit']

        # def create(self, validated_data):
        #     return LoadCategoryModel.objects.create(**validated_data)


class vehicleCategoryModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = vehicleCategoryModel
        fields = ['id', 'n', 'f', 'c', 'p', 'e', 'r',
                  'k', 'l', 'g', 'h', 's', 'u', 'CAGR_V', 'baseElectricityTariff']


class EvAnalysisSerializer(serializers.ModelSerializer):
    # isManagedFile = serializers.SerializerMethodField('get_image')

    # def get_image(self, obj):
    #     return self.context['request'].build_absolute_uri(obj.isManagedFile_upload.url)

    class Meta:
        model = evAnalysis
        fields = ['id', 'loadCategory', 'isLoadSplit', 'loadCategory1', 'loadCategory2',  'loadCategory3', 'loadCategory4', 'loadCategory5',  'loadCategory6', 'isLoadSplitFile', 'numOfvehicleCategory', 'vehicleCategoryData1',
                  'vehicleCategoryData2', 'vehicleCategoryData3', 'vehicleCategoryData4', 'vehicleCategoryData5', 'resolution', 'BR_F', 'sharedSavaing', 'sum_pk_cost', 'sum_zero_cost', 'sum_op_cost', 'win_pk_cost', 'win_zero_cost', 'win_op_cost', 'date1_start', 'date1_end', 'date2_start', 'date2_end', 's_pks', 's_pke', 's_sx', 's_ops', 's_ope', 's_rb', 'w_pks', 'w_pke', 'w_sx', 'w_ops', 'w_ope', 'w_rb', 'fileId','user_name']


class FilesSerializer(serializers.ModelSerializer):
    class Meta():
        model = Files
        fields = ['file', 'id']
