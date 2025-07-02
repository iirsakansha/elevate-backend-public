from django.urls import path
from . import views
from knox import views as knox_views

app_name = 'elevate'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/get-csrf-token/', views.get_csrf_token, name='get_csrf_token'),
    path('api/register/', views.RegisterAPI.as_view(), name='register'),
    path('api/login/', views.LoginAPI.as_view(), name='login'),
    path('api/logout/', knox_views.LogoutView.as_view(), name='logout'),
    path('api/change-password/', views.ChangePasswordView.as_view(),
         name='change_password'),
    path('api/invite-user/', views.InviteUserAPI.as_view(), name='invite_user'),
    path('api/user-profile/', views.InvitedUserProfileAPI.as_view(),
         name='user_profile'),
    path('api/set-password/', views.SetPasswordAPI.as_view(), name='set_password'),
    path('api/password-reset-request/',
         views.PasswordResetRequestAPI.as_view(), name='password_reset_request'),
    path('api/password-reset-confirm/<uidb64>/<token>/',
         views.password_reset_confirm, name='password_reset_confirm'),
    path('api/password-reset/', views.PasswordResetAPI.as_view(),
         name='password_reset'),
    path('api/password-reset/validate/',
         views.PasswordResetLinkValidateAPI.as_view(), name='password_reset_validate'),
    path('api/invitation/<str:token>/',
         views.GetInvitationDetailsAPI.as_view(), name='get_invitation_details'),
    path('api/invited-users/all/', views.ListInvitedUsersAPI.as_view(),
         name='list_invited_users'),
    path('api/delete-user/<int:id>/',
         views.DeleteInvitedUserAPI.as_view(), name='delete_invited_user'),
    path('api/analyses/', views.AnalysisListCreateAPI.as_view(),
         name='analysis_list_create'),
    path('api/analyses/<int:id>/', views.AnalysisRetrieveUpdateDestroyAPI.as_view(),
         name='analysis_retrieve_update_destroy'),
    path('api/ev-analysis/', views.AnalysisView.as_view(), name='ev_analysis'),
    path('api/delete-data/', views.DeleteDataAPI.as_view(), name='delete_data'),
    path('api/file-upload/', views.FileUploadAPI.as_view(), name='file_upload'),
    path('api/user-analysis-log/', views.UserAnalysisLogAPI.as_view(),
         name='user_analysis_log'),
    path('api/permanent-analyses/', views.PermanentAnalysisListCreateAPI.as_view(),
         name='permanent_analysis_list_create'),
    path('api/permanent-analyses/<int:pk>/', views.PermanentAnalysisRetrieveUpdateDestroyAPI.as_view(),
         name='permanent_analysis_retrieve_update_destroy'),


]
