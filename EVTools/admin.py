from knox import models
from django.contrib import auth
import django.contrib.auth.models
from django.contrib.auth.models import User
from django.contrib import admin
from .models import evAnalysis, LoadCategoryModel, vehicleCategoryModel, Files, userAnalysis

admin.site.site_header = 'EV Tool - Admin'
admin.site.site_title = 'EV Tool - Admin'

# Register your models here.

# admin.site.unregister(models.AuthToken)


class AdminuserAnalysis(admin.ModelAdmin):
    list_display = ['status', 'id', 'userName', 'errorLog', 'time']


admin.site.register(evAnalysis)
# admin.site.register(LoadCategoryModel)
# admin.site.register(vehicleCategoryModel)
# admin.site.register(Files)
admin.site.register(userAnalysis, AdminuserAnalysis)


