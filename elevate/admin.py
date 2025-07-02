from django.contrib import admin
from .models import UserProfile, Analysis, LoadCategoryModel, VehicleCategoryModel, Files, UserAnalysis

admin.site.site_header = 'EV Tool - Admin'
admin.site.site_title = 'EV Tool - Admin'


class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'organization', 'invitation_status', 'created_at']


class AnalysisAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'created_at', 'updated_at']


class UserAnalysisAdmin(admin.ModelAdmin):
    list_display = ['userName', 'status', 'time']


admin.site.register(UserProfile, UserProfileAdmin)
admin.site.register(Analysis, AnalysisAdmin)
admin.site.register(LoadCategoryModel)
admin.site.register(VehicleCategoryModel)
admin.site.register(Files)
admin.site.register(UserAnalysis, UserAnalysisAdmin)
