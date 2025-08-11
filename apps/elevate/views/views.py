# elevate/views.py
import logging
import os
import shutil
from datetime import datetime

from django.conf import settings
from django.contrib.auth.models import User
from django.core.mail import EmailMessage
from django.db import transaction
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils.crypto import get_random_string
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.timezone import now
from django.views.decorators.csrf import ensure_csrf_cookie
from knox.models import AuthToken
from rest_framework import generics, permissions, status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from ..exceptions import (
    AnalysisProcessingError,
    InvalidCategoryError,
    InvalidDateError,
    InvalidFileError,
    InvalidTimeFormatError,
)
from ..models.models import Analysis, PermanentAnalysis, UserAnalysis, UserProfile
from ..serializers.serializers import (
    AnalysisSerializer,
    ChangePasswordSerializer,
    FilesSerializer,
    InvitedUserProfileSerializer,
    LoginUserSerializer,
    PasswordResetNoEmailSerializer,
    PasswordResetSerializer,
    PermanentAnalysisSerializer,
    SetPasswordSerializer,
    UserAnalysisSerializer,
    UserSerializer,
)
from ..services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

# Constants (consider moving to constants.py)
ALLOWED_PASSWORD_RESET_ROLES = ["admin", "superuser"]
UPLOAD_FOLDER = "file-upload"


@ensure_csrf_cookie
def get_csrf_token(request):
    """Set CSRF cookie for the request."""
    return HttpResponse("CSRF cookie set")


def index(request):
    """Render the index page."""
    return render(request, "templates/elevate/index.html", {})


class RegisterAPI(APIView):
    permission_classes = [AllowAny]
    """API for registering new users directly (not invited)."""

    def post(self, request):
        email = request.data.get("email", "").lower().strip()
        if not email:
            return Response(
                {"error": "Email is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        if User.objects.filter(email=email).exists():
            return Response(
                {"error": "User with this email already exists"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        random_password = get_random_string(length=12)
        username = (
            email.split("@")[0] + get_random_string(4).lower()
        )  # ensure uniqueness

        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=random_password,
                    is_staff=False,
                )
                UserProfile.objects.create(
                    user=user,
                    organization="World Resources Institute",
                    invitation_status="Completed",
                    invitation_username=username,
                    temporary_password=random_password,
                    role="user",
                    is_self_registered=True,
                )

                context = {
                    "username": username,
                    "random_password": random_password,
                    "current_year": datetime.now().year,
                    "signin_url": f"{settings.FRONTEND_URL}/signin",
                }

                html_message = render_to_string("emails/self-register.html", context)
                email_msg = EmailMessage(
                    subject="Your Elevate Account Credentials",
                    body=html_message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[email],
                )
                email_msg.content_subtype = "html"
                email_msg.send()

            return Response(
                {"message": "Registration successful. Credentials sent to email."},
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            logger.error(f"Registration email sending failed: {str(e)}")
            return Response(
                {"error": "Failed to register", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class LoginAPI(generics.GenericAPIView):
    """API for user login."""

    serializer_class = LoginUserSerializer
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        logger.info(f"Login attempt for {request.data.get('username', 'unknown')}")
        if not request.data:
            return Response(
                {"error": "Request body is empty"}, status=status.HTTP_400_BAD_REQUEST
            )
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Login failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        user = serializer.validated_data
        try:
            if hasattr(user, "profile") and user.profile.invitation_status == "Pending":
                user.profile.invitation_status = "Accepted"
                user.profile.save()
            response_data = {
                "user": UserSerializer(
                    user, context=self.get_serializer_context()
                ).data,
                "token": AuthToken.objects.create(user)[1],
                "message": "Login successful",
            }
            logger.info(f"Login successful for {user.username}")
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return Response(
                {"error": "An error occurred during login"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ChangePasswordView(generics.UpdateAPIView):
    """API to change user password."""

    serializer_class = ChangePasswordSerializer
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        if request.user.profile.is_self_registered:
            return Response(
                {"error": "You are not allowed to change password."}, status=403
            )

    def get_object(self):
        return self.request.user

    def update(self, request, *args, **kwargs):
        user = self.get_object()
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            if not user.check_password(serializer.validated_data.get("old_password")):
                return Response(
                    {"error": "Invalid current password"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            user.set_password(serializer.validated_data.get("new_password"))
            user.save()
            return Response(
                {"message": "Password updated successfully"}, status=status.HTTP_200_OK
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class InviteUserAPI(APIView):
    """API for inviting new users."""

    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        email = request.data.get("email", "").lower().strip()
        username = request.data.get("username", "").strip()
        organization = request.data.get("organization", "World Resources Institute")
        if not email or not username:
            return Response(
                {"error": "Both email and username are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if (
            User.objects.filter(email=email).exists()
            or User.objects.filter(username=username).exists()
        ):
            return Response(
                {"error": "User with this email or username already exists"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        random_password = get_random_string(length=12)
        token = get_random_string(length=32)
        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=random_password,
                    is_staff=False,
                )
                UserProfile.objects.create(
                    user=user,
                    organization=organization,
                    invitation_status="Pending",
                    invitation_username=username,
                    temporary_password=random_password,
                    invitation_token=token,
                    role="user",
                )
                activation_link = f"{settings.FRONTEND_URL}/signin/{token}"
                context = {
                    "username": username,
                    "admin_name": request.user.get_full_name() or request.user.username,
                    "random_password": random_password,
                    "activation_link": activation_link,
                    "current_year": datetime.now().year,
                }
                html_message = render_to_string(
                    "emails/account-activation.html", context
                )
                email = EmailMessage(
                    subject="Your Elevate Account Activation",
                    body=html_message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[email],
                )
                email.content_subtype = "html"
                email.send()
            return Response(
                {"message": "Invitation sent successfully"},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            return Response(
                {"error": "Failed to send invitation email", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class InvitedUserProfileAPI(generics.RetrieveUpdateAPIView):
    """API to retrieve or update invited user profile."""

    serializer_class = InvitedUserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


class SetPasswordAPI(generics.GenericAPIView):
    """API for invited users to set their password."""

    serializer_class = SetPasswordSerializer
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        if request.user.is_staff:
            return Response(
                {"error": "Admins cannot use this endpoint to set passwords"},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = request.user
        if user.profile.invitation_status != "Pending":
            return Response(
                {"error": "This account is already activated"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        user.set_password(serializer.validated_data["new_password"])
        user.profile.invitation_status = "Accepted"
        user.profile.temporary_password = ""
        user.profile.invitation_token = ""
        user.profile.save()
        user.save()
        return Response(
            {"message": "Password set successfully"}, status=status.HTTP_200_OK
        )


class PasswordResetRequestAPI(APIView):
    """API to request a password reset."""

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PasswordResetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data["email"].lower().strip()
        try:
            user = User.objects.get(email=email)
            if user.is_superuser and not hasattr(user, "profile"):
                pass
            else:
                if not hasattr(user, "profile"):
                    return Response(
                        {"error": "User profile is missing"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                if user.profile.invitation_status.lower() == "pending":
                    return Response(
                        {"error": "Invited users cannot request password reset"},
                        status=status.HTTP_403_FORBIDDEN,
                    )
                if user.profile.role.lower() not in ALLOWED_PASSWORD_RESET_ROLES:
                    return Response(
                        {"error": "Only admin users can request password reset"},
                        status=status.HTTP_403_FORBIDDEN,
                    )
            token = get_random_string(length=32)
            if hasattr(user, "profile"):
                user.profile.password_reset_token = token
                user.profile.save()
            frontend_url = getattr(settings, "FRONTEND_URL", "http://localhost:5173")
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            reset_link = f"{frontend_url}/password-reset/{uid}/{token}/"
            context = {"reset_link": reset_link, "current_year": now().year}
            html_message = render_to_string("emails/password-reset.html", context)
            email_message = EmailMessage(
                subject="Elevate Password Reset",
                body=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[email],
            )
            email_message.content_subtype = "html"
            email_message.send()
            return Response(
                {"message": "Password reset email sent"}, status=status.HTTP_200_OK
            )
        except User.DoesNotExist:
            return Response(
                {"error": "User not found"}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Failed to send password reset email: {str(e)}")
            return Response(
                {"error": "Failed to send password reset email"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class PasswordResetAPI(generics.GenericAPIView):
    """API to reset password using a token."""

    serializer_class = PasswordResetNoEmailSerializer
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        uidb64 = serializer.validated_data["uidb64"]
        token = serializer.validated_data["token"]
        new_password = serializer.validated_data["new_password"]
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
            if not user.is_staff:
                return Response(
                    {"error": "Password reset is only allowed for admin users"},
                    status=status.HTTP_403_FORBIDDEN,
                )
            if user.is_superuser and not hasattr(user, "profile"):
                user.set_password(new_password)
                user.save()
                return Response(
                    {"message": "Password reset successfully"},
                    status=status.HTTP_200_OK,
                )
            if hasattr(user, "profile") and user.profile.password_reset_token == token:
                user.set_password(new_password)
                user.profile.password_reset_token = ""
                user.profile.save()
                user.save()
                return Response(
                    {"message": "Password reset successfully"},
                    status=status.HTTP_200_OK,
                )
            return Response(
                {"error": "Invalid token"}, status=status.HTTP_400_BAD_REQUEST
            )
        except (User.DoesNotExist, ValueError):
            return Response(
                {"error": "Invalid user or token"}, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Unexpected error in password reset: {str(e)}")
            return Response(
                {"error": f"Unexpected error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


def password_reset_confirm(request, uidb64, token):
    """Confirm password reset link and redirect to frontend."""
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
        if not user.is_staff:
            return redirect(
                f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')}/password-reset/invalid/"
            )
        if user.profile.password_reset_token == token:
            return redirect(
                f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')}/password-reset/{uidb64}/{token}/"
            )
        return redirect(
            f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')}/password-reset/invalid/"
        )
    except (User.DoesNotExist, ValueError):
        return redirect(
            f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')}/password-reset/invalid/"
        )


class PasswordResetLinkValidateAPI(APIView):
    """API to validate password reset link."""

    permission_classes = [AllowAny]

    def get(self, request, uidb64, token):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
            if user.is_superuser and not hasattr(user, "profile"):
                return Response({"valid": True}, status=status.HTTP_200_OK)
            if hasattr(user, "profile"):
                stored_token = user.profile.password_reset_token
                if stored_token == token:
                    return Response({"valid": True}, status=status.HTTP_200_OK)
                return Response(
                    {"valid": False, "error": "Invalid or expired token"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            return Response(
                {"valid": False, "error": "User is not allowed to reset password"},
                status=status.HTTP_403_FORBIDDEN,
            )
        except User.DoesNotExist:
            return Response(
                {"valid": False, "error": "User does not exist"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.error(f"Unexpected error in link validation: {str(e)}")
            return Response(
                {"valid": False, "error": f"Unexpected error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class GetInvitationDetailsAPI(APIView):
    """API to retrieve invitation details by token."""

    permission_classes = [AllowAny]

    def get(self, request, token):
        try:
            profile = UserProfile.objects.get(invitation_token=token)
            return Response(
                {
                    "username": profile.invitation_username,
                    "email": profile.user.email,
                    "temporary_password": profile.temporary_password,
                    "organization": profile.organization,
                },
                status=status.HTTP_200_OK,
            )
        except UserProfile.DoesNotExist:
            return Response(
                {"error": "Invalid invitation token"}, status=status.HTTP_404_NOT_FOUND
            )


class ListInvitedUsersAPI(generics.ListAPIView):
    """API to list invited users and total number of templates."""

    serializer_class = InvitedUserProfileSerializer
    permission_classes = [permissions.IsAdminUser]

    def get_queryset(self):
        return User.objects.filter(
            Q(profile__invitation_status__in=["Pending", "Accepted"])
            | Q(profile__is_self_registered=True)
        )

    def list(self, request, *args, **kwargs):
        """Override list method to include total template count."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        total_templates = PermanentAnalysis.objects.count()
        response_data = {
            "invited_users": serializer.data,
            "total_templates": total_templates,
        }
        return Response(response_data)


class DeleteInvitedUserAPI(generics.DestroyAPIView):
    """API to delete an invited user."""

    permission_classes = [permissions.IsAdminUser]
    queryset = User.objects.all()
    lookup_field = "id"

    def perform_destroy(self, instance):
        instance.profile.delete()
        instance.delete()


class AnalysisView(APIView):
    """API to perform EV analysis."""

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def post(self, request, *args, **kwargs):
        """Process EV analysis with file upload and category data."""
        analysis_service = AnalysisService(request)
        try:
            results = analysis_service.process_analysis()
            return Response(results, status=status.HTTP_201_CREATED)
        except (
            InvalidFileError,
            InvalidCategoryError,
            InvalidDateError,
            InvalidTimeFormatError,
        ) as e:
            logger.error(f"Analysis failed: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except AnalysisProcessingError as e:
            logger.error(f"Analysis processing error: {str(e)}")
            return Response(
                {"error": str(e), "traceback": e.traceback},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        except Exception as e:
            logger.error(f"Unexpected error in analysis: {str(e)}")
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AnalysisListCreateAPI(generics.ListCreateAPIView):
    """API to list or create analyses."""

    queryset = Analysis.objects.all()
    serializer_class = AnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if self.request.user.is_staff:
            return Analysis.objects.all()
        return Analysis.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class AnalysisRetrieveUpdateDestroyAPI(generics.RetrieveUpdateDestroyAPIView):
    """API to retrieve, update, or delete an analysis."""

    queryset = Analysis.objects.all()
    serializer_class = AnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]
    lookup_field = "id"

    def get_queryset(self):
        if self.request.user.is_staff:
            return Analysis.objects.all()
        return Analysis.objects.filter(user=self.request.user)

    def perform_update(self, serializer):
        serializer.save(user=self.request.user)


class FileUploadAPI(APIView):
    """API to upload files."""

    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        file_serializer = FilesSerializer(
            data=request.data, context={"request": request}
        )
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DeleteDataAPI(APIView):
    """API to delete analysis data and associated files."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        folder_id = request.data.get("folder_id")
        try:
            is_folder = os.path.isdir(
                os.path.join(settings.MEDIA_ROOT, "outputs", str(folder_id))
            )
            if is_folder:
                shutil.rmtree(
                    os.path.join(settings.MEDIA_ROOT, "outputs", str(folder_id)),
                    ignore_errors=True,
                )
            analysis = Analysis.objects.get(id=folder_id, user=self.request.user)
            for load_cat in analysis.load_categories.all():
                load_cat.delete()
            for vehicle_cat in analysis.vehicle_categories.all():
                vehicle_cat.delete()
            analysis.delete()
            return Response(
                {"message": "Data deleted successfully"}, status=status.HTTP_200_OK
            )
        except Analysis.DoesNotExist:
            return Response(
                {"message": "Analysis not found, folder deleted if existed"},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            logger.error(f"Error deleting analysis: {str(e)}")
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class UserAnalysisLogAPI(generics.ListAPIView):
    """API to list user analysis logs."""

    serializer_class = UserAnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if self.request.user.is_staff:
            return UserAnalysis.objects.all()
        return UserAnalysis.objects.filter(user_name=self.request.user.username)


class IsAdminUser(permissions.BasePermission):
    """Custom permission to check if user is an admin."""

    def has_permission(self, request, view):
        return request.user and request.user.is_staff


class PermanentAnalysisListCreateAPI(generics.ListCreateAPIView):
    """API to list or create permanent analyses."""

    queryset = PermanentAnalysis.objects.all()
    serializer_class = PermanentAnalysisSerializer

    def get_permissions(self):
        """Return permissions based on request method."""
        if self.request.method == "POST":
            return [permissions.IsAdminUser()]
        return [permissions.AllowAny()]


class PermanentAnalysisRetrieveUpdateDestroyAPI(generics.RetrieveUpdateDestroyAPIView):
    """API to retrieve, update, or delete permanent analyses."""

    queryset = PermanentAnalysis.objects.all()
    serializer_class = PermanentAnalysisSerializer
    permission_classes = [permissions.AllowAny]
