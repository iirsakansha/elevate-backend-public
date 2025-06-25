from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import get_user_model
from django.contrib.auth.models import User
from django.core.mail import EmailMessage, send_mail
from django.template.loader import render_to_string
from django.utils.crypto import get_random_string
from django.utils.encoding import force_str, force_bytes
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.conf import settings
from rest_framework import generics, status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from knox.models import AuthToken

from EVTools.serializers import LoginUserSerializer
from EVTools.serializers import UserSerializer
from .models import UserProfile
from .serializers import PasswordResetNoEmailSerializer, SetPasswordSerializer, InvitedUserProfileSerializer
from .utils import account_activation_token
from urllib.parse import quote
import logging
from datetime import datetime
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.tokens import default_token_generator

logger = logging.getLogger(__name__)


@ensure_csrf_cookie
def get_csrf_token(request):
    return HttpResponse("CSRF cookie set")


class InviteUserAPI(APIView):
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        email = request.data.get('email', '').lower().strip()
        username = request.data.get('username', '').strip()
        organization = request.data.get(
            'organization', 'World Resources Institute')

        if not email or not username:
            return Response(
                {"error": "Both email and username are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if User.objects.filter(email=email).exists() or User.objects.filter(username=username).exists():
            return Response(
                {"error": "User with this email or username already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )

        random_password = get_random_string(length=12)
        token = get_random_string(length=32)
        user = User.objects.create_user(
            username=username,
            email=email,
            password=random_password,
            is_staff=True
        )

        UserProfile.objects.create(
            user=user,
            organization=organization,
            invitation_status='Pending',
            invitation_username=username,
            temporary_password=random_password,
            invitation_token=token
        )

        activation_link = f"http://localhost:3006/signin/{quote(token)}"

        context = {
            'username': username,
            'admin_name': request.user.get_full_name() or request.user.username,
            'random_password': random_password,
            'activation_link': activation_link,
            'current_year': datetime.now().year,
        }
        logger.info(f"Email context: {context}")

        try:
            html_message = render_to_string(
                'emails/account_activation.html', context)
            logger.info(f"Rendered email HTML: {html_message}")
            email = EmailMessage(
                subject='Your EVTools Admin Account Activation',
                body=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[email],
            )
            email.content_subtype = "html"
            email.send()

            return Response({
                "message": "Admin invitation sent successfully",
                "debug_link": activation_link if settings.DEBUG else None
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            user.delete()
            logger.error(f"Email sending failed: {str(e)}")
            return Response(
                {"error": "Failed to send invitation email",
                 "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AdminLoginAPI(generics.GenericAPIView):
    serializer_class = LoginUserSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data

        if user.is_superuser:
            return Response({
                "user": UserSerializer(user, context=self.get_serializer_context()).data,
                "token": AuthToken.objects.create(user)[1]
            })

        try:
            user_profile = UserProfile.objects.get(user=user)
        except UserProfile.DoesNotExist:
            return Response(
                {"error": "User not found in invitation records"},
                status=status.HTTP_403_FORBIDDEN
            )

        if user_profile.invitation_status != 'Pending':
            return Response(
                {"error": "Only superusers or invited admins can log in"},
                status=status.HTTP_403_FORBIDDEN
            )

        user_profile.invitation_status = 'Accepted'
        user_profile.save()

        return Response({
            "user": UserSerializer(user, context=self.get_serializer_context()).data,
            "token": AuthToken.objects.create(user)[1]
        })


class GetInvitationDetailsAPI(APIView):
    def get(self, request, token):
        try:
            user_profile = UserProfile.objects.get(invitation_token=token)
            if user_profile.invitation_status != 'Pending':
                return Response(
                    {"error": "Invitation is no longer valid"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            return Response({
                "username": user_profile.invitation_username,
                "password": user_profile.temporary_password
            }, status=status.HTTP_200_OK)
        except UserProfile.DoesNotExist:
            return Response(
                {"error": "Invalid or expired token"},
                status=status.HTTP_404_NOT_FOUND
            )


class SetPasswordAPI(APIView):
    permission_classes = [IsAuthenticated]  # Require authentication

    def put(self, request):
        serializer = SetPasswordSerializer(
            data=request.data, context={'request': request})
        if serializer.is_valid():
            user = request.user
            user.set_password(serializer.validated_data['new_password'])
            user.save()
            return Response({"message": "Password changed successfully"}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PasswordResetRequestAPI(APIView):
    def post(self, request):
        email = request.data.get('email')
        try:
            user = User.objects.get(email=email)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = account_activation_token.make_token(user)

            reset_link = request.build_absolute_uri(
                reverse('admin_panel:password_reset_confirm', kwargs={  # Add namespace
                    'uidb64': uid, 'token': token})
            )

            context = {
                'username': user.username,
                'reset_link': reset_link,
                'current_year': datetime.now().year,
            }
            html_message = render_to_string(
                'emails/password_reset.html', context)

            subject = 'Reset Your EVTools Password'
            email_message = EmailMessage(
                subject=subject,
                body=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[email],
            )
            email_message.content_subtype = "html"
            email_message.send(fail_silently=False)

            return Response({"message": "Password reset email sent"}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Failed to send password reset email: {str(e)}")
            return Response(
                {"error": "Failed to send password reset email",
                    "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

def password_reset_confirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and account_activation_token.check_token(user, token):
        # Redirect to frontend with uidb64 and token as query parameters
        redirect_url = f'http://localhost:5173/reset-password?uidb64={uidb64}&token={token}'
        return redirect(redirect_url)
    else:
        return HttpResponse('Password reset link is invalid!', status=400)
    

class PasswordResetAPI(APIView):
    def post(self, request):
        serializer = PasswordResetNoEmailSerializer(data=request.data)
        if serializer.is_valid():
            new_password = serializer.validated_data['new_password']
            uidb64 = serializer.validated_data['uidb64']
            token = serializer.validated_data['token']

            try:
                uid = force_str(urlsafe_base64_decode(uidb64))
                user = User.objects.get(pk=uid)
                if not account_activation_token.check_token(user, token):
                    return Response({"error": "Invalid or expired reset token"}, status=400)
                user.set_password(new_password)
                user.save()
                return Response({"message": "Password reset successfully"}, status=200)
            except User.DoesNotExist:
                return Response({"error": "User not found"}, status=404)
        return Response(serializer.errors, status=400)

class PasswordResetLinkValidateAPI(APIView):
    def get(self, request):
        uidb64 = request.query_params.get('uidb64')
        token = request.query_params.get('token')

        if not uidb64 or not token:
            return Response({"valid": False}, status=status.HTTP_400_BAD_REQUEST)

        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            return Response({"valid": False}, status=status.HTTP_400_BAD_REQUEST)

        if account_activation_token.check_token(user, token):
            return Response({"valid": True}, status=status.HTTP_200_OK)
        else:
            return Response({"valid": False}, status=status.HTTP_400_BAD_REQUEST)


class InvitedUserProfileAPI(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        try:
            serializer = InvitedUserProfileSerializer(user)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except UserProfile.DoesNotExist:
            return Response({"error": "User profile not found"}, status=status.HTTP_404_NOT_FOUND)


class ListInvitedUsersAPI(generics.ListAPIView):
    permission_classes = [permissions.IsAdminUser]
    serializer_class = InvitedUserProfileSerializer
    queryset = User.objects.filter(profile__invitation_status__in=[
                                   'Pending', 'Accepted']).select_related('profile')

    def list(self, request, *args, **kwargs):
        try:
            queryset = self.filter_queryset(self.get_queryset())
            total_users = queryset.count()  # Get the total number of users
            serializer = self.get_serializer(queryset, many=True)
            response_data = {
                "total_users": total_users,
                "users": serializer.data
            }
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error fetching invited users: {str(e)}")
            return Response(
                {"error": "Failed to fetch invited users", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DeleteInvitedUserAPI(APIView):
    permission_classes = [permissions.IsAdminUser]

    def delete(self, request, id):
        try:
            user = User.objects.get(id=id, profile__invitation_status__in=[
                                    'Pending', 'Accepted'])
            username = user.username
            user.delete()  # This will also delete the associated UserProfile due to CASCADE
            logger.info(
                f"User {username} (ID: {id}) deleted successfully by {request.user.username}")
            return Response(
                {"message": "User deleted successfully"},
                status=status.HTTP_200_OK
            )
        except User.DoesNotExist:
            logger.warning(
                f"Attempt to delete non-existent user ID: {id} by {request.user.username}")
            return Response(
                {"error": "User not found or not an invited user"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(
                f"Error deleting user ID: {id} by {request.user.username}: {str(e)}")
            return Response(
                {"error": "Failed to delete user", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
