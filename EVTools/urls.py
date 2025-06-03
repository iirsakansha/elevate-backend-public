from django.contrib.auth.decorators import login_required
from django.urls import path
from . import views
from EVTools.views import RegisterAPI, LoginAPI, ChangePasswordView, UserAPI, EvAnalysisView, DeleteData, FileUpload
from knox import views as knox_views
EVTools = "EVTools"

urlpatterns = [
    path('', views.index, name='index'),
    path('api/login/', LoginAPI.as_view(), name='login'),
    path('api/logout/',  knox_views.LogoutView.as_view(), name='knox_logout'),
    path('api/change-password/', ChangePasswordView.as_view(),
         name='change-password'),
    path('api/user/', UserAPI.as_view(), name='user'),
    path('api/ev-analysis/', EvAnalysisView.as_view(), name='ev-analysis'),
    path('api/delete-data/', DeleteData.as_view(), name='delete-data'),
    path('api/file-upload/', FileUpload.as_view(), name='file-upload'),
]
