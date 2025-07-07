# elevate/admin.py
from django.contrib import admin
from .models import UserProfile, LoadCategoryModel, VehicleCategoryModel, Analysis, PermanentAnalysis, Files, UserAnalysis


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'organization', 'invitation_status', 'role']
    search_fields = ['user__username', 'organization']
    list_filter = ['invitation_status', 'role']


@admin.register(LoadCategoryModel)
class LoadCategoryModelAdmin(admin.ModelAdmin):
    list_display = ['category', 'sales_cagr', 'specify_split']
    search_fields = ['category']
    list_filter = ['category']


@admin.register(VehicleCategoryModel)
class VehicleCategoryModelAdmin(admin.ModelAdmin):
    list_display = ['vehicle_category', 'vehicle_count', 'cagr_v']
    search_fields = ['vehicle_category']
    list_filter = ['vehicle_category']


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'created_at', 'updated_at']
    search_fields = ['name', 'user__username']
    list_filter = ['created_at', 'user']


@admin.register(PermanentAnalysis)
class PermanentAnalysisAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'created_at', 'updated_at']
    search_fields = ['name', 'user__username']
    list_filter = ['created_at', 'user']


@admin.register(Files)
class FilesAdmin(admin.ModelAdmin):
    list_display = ['id', 'file']
    search_fields = ['file']


@admin.register(UserAnalysis)
class UserAnalysisAdmin(admin.ModelAdmin):
    # Updated from 'userName' to 'user_name'
    list_display = ['user_name', 'status', 'time']
    search_fields = ['user_name']
    list_filter = ['status', 'time']
