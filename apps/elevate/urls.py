from django.urls import path
from knox import views as knox_views

from .views import views

app_name = "elevate"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/register/", views.RegisterAPI.as_view(), name="register"),
    path("api/get-csrf-token/", views.get_csrf_token, name="get-csrf-token"),
    path("api/login/", views.LoginAPI.as_view(), name="login"),
    path("api/logout/", knox_views.LogoutView.as_view(), name="logout"),
    path(
        "api/change-password/",
        views.ChangePasswordView.as_view(),
        name="change-password",
    ),
    path("api/invite-user/", views.InviteUserAPI.as_view(), name="invite-user"),
    path(
        "api/user-profile/", views.InvitedUserProfileAPI.as_view(), name="user-profile"
    ),
    path("api/set-password/", views.SetPasswordAPI.as_view(), name="set-password"),
    path(
        "api/password-reset-request/",
        views.PasswordResetRequestAPI.as_view(),
        name="password-reset-request",
    ),
    path(
        "api/password-reset-confirm/<uidb64>/<token>/",
        views.password_reset_confirm,
        name="password-reset-confirm",
    ),
    path(
        "api/password-reset/", views.PasswordResetAPI.as_view(), name="password-reset"
    ),
    path(
        "api/password-reset/validate/<str:uidb64>/<str:token>/",
        views.PasswordResetLinkValidateAPI.as_view(),
        name="password-reset-validate",
    ),
    path(
        "api/invitation/<str:token>/",
        views.GetInvitationDetailsAPI.as_view(),
        name="get-invitation-details",
    ),
    path(
        "api/invited-users/all/",
        views.ListInvitedUsersAPI.as_view(),
        name="list-invited-users",
    ),
    path(
        "api/delete-user/<int:id>/",
        views.DeleteInvitedUserAPI.as_view(),
        name="delete-invited-user",
    ),
    path(
        "api/analyses/",
        views.AnalysisListCreateAPI.as_view(),
        name="analysis-list-create",
    ),
    path(
        "api/analyses/<int:id>/",
        views.AnalysisRetrieveUpdateDestroyAPI.as_view(),
        name="analysis-retrieve-update-destroy",
    ),
    path("api/ev-analysis/", views.AnalysisView.as_view(), name="ev-analysis"),
    path("api/delete-data/", views.DeleteDataAPI.as_view(), name="delete-data"),
    path("api/file-upload/", views.FileUploadAPI.as_view(), name="file-upload"),
    path(
        "api/user-analysis-log/",
        views.UserAnalysisLogAPI.as_view(),
        name="user-analysis-log",
    ),
    path(
        "api/permanent-analyses/",
        views.PermanentAnalysisListCreateAPI.as_view(),
        name="permanent-analysis-list-create",
    ),
    path(
        "api/permanent-analyses/<int:pk>/",
        views.PermanentAnalysisRetrieveUpdateDestroyAPI.as_view(),
        name="permanent-analysis-retrieve-update-destroy",
    ),
]
