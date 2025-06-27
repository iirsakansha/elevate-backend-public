from django.urls import path
from . import views

app_name = 'admin_panel'

urlpatterns = [
    path('get-csrf-token/', views.get_csrf_token, name='get_csrf_token'),
    path('api/invite-user/', views.InviteUserAPI.as_view(), name='invite_user'),
    path('api/admin-profile/', views.InvitedUserProfileAPI.as_view(),
         name='invited_user_profile'),
    path('api/set-password/', views.SetPasswordAPI.as_view(), name='set_password'),
    path('api/password-reset-request/',
         views.PasswordResetRequestAPI.as_view(), name='password_reset_request'),
    path('api/password-reset-confirm/<uidb64>/<token>/',
         views.password_reset_confirm, name='password_reset_confirm'),
    path('api/password-reset/', views.PasswordResetAPI.as_view(),
         name='password_reset'),
    path('api/password-reset/validate/',
         views.PasswordResetLinkValidateAPI.as_view(), name='password_reset_validate'),
    path('admin/invite-user/', views.InviteUserAPI.as_view(),
         name='admin_invite_user'),
    path('api/admin-login/', views.AdminLoginAPI.as_view(), name='admin_login'),
    path('api/invitation/<str:token>/',
         views.GetInvitationDetailsAPI.as_view(), name='get_invitation_details'),
    path('api/invited-users/all/', views.ListInvitedUsersAPI.as_view(),
         name='list_invited_users'),
    path('api/delete-user/<int:id>/',
         views.DeleteInvitedUserAPI.as_view(), name='delete_invited_user'),
    path('api/templates/', views.TemplateListCreateAPI.as_view(),
         name='template_list_create'),
    path('api/templates/<int:id>/', views.TemplateRetrieveUpdateDestroyAPI.as_view(),
         name='template_retrieve_update_destroy')
]
