from urllib import request
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.core.mail import EmailMessage
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
from knox.views import LogoutView as KnoxLogoutView
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.core.files.storage import FileSystemStorage
from .models import PermanentAnalysis, UserProfile, Analysis, LoadCategoryModel, VehicleCategoryModel, Files, UserAnalysis
from .serializers import (
    PermanentAnalysisSerializer, RegisterSerializer, LoginUserSerializer, ChangePasswordSerializer, SetPasswordSerializer,
    PasswordResetSerializer, PasswordResetNoEmailSerializer, InvitedUserProfileSerializer,
    UserSerializer, AnalysisSerializer, LoadCategoryModelSerializer, VehicleCategoryModelSerializer,
    FilesSerializer, UserAnalysisSerializer
)
import logging
import os
import shutil
import numpy as np
import pandas as pd
from scipy import stats as ss
import math
import time
import traceback
from datetime import datetime
from django.db import transaction

logger = logging.getLogger(__name__)


@ensure_csrf_cookie
def get_csrf_token(request):
    return HttpResponse("CSRF cookie set")


def index(request):
    return render(request, 'elevate/index.html', {})


class RegisterAPI(generics.GenericAPIView):
    serializer_class = RegisterSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({
            "user": UserSerializer(user, context=self.get_serializer_context()).data,
            "token": AuthToken.objects.create(user)[1]
        }, status=status.HTTP_201_CREATED)


class LoginAPI(generics.GenericAPIView):
    serializer_class = LoginUserSerializer
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        logger.info(
            f"Login attempt for {request.data.get('username', 'unknown')}")
        if not request.data:
            return Response({"error": "Request body is empty"}, status=status.HTTP_400_BAD_REQUEST)
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Login failed: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        user = serializer.validated_data
        try:
            if hasattr(user, 'profile') and user.profile.invitation_status == 'Pending':
                user.profile.invitation_status = 'Accepted'
                user.profile.save()
            response_data = {
                "user": UserSerializer(user, context=self.get_serializer_context()).data,
                "token": AuthToken.objects.create(user)[1]
            }
            logger.info(f"Login successful for {user.username}")
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return Response({"error": "An error occurred during login"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChangePasswordView(generics.UpdateAPIView):
    serializer_class = ChangePasswordSerializer
    permission_classes = (permissions.IsAuthenticated,)

    def get_object(self, queryset=None):
        return self.request.user

    def update(self, request, *args, **kwargs):
        self.object = self.get_object()
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            if not self.object.check_password(serializer.data.get("old_password")):
                return Response({"error": "Invalid current password"}, status=status.HTTP_400_BAD_REQUEST)
            self.object.set_password(serializer.data.get("new_password"))
            self.object.save()
            return Response({"message": "Password updated successfully"}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class InviteUserAPI(APIView):
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
        email = request.data.get('email', '').lower().strip()
        username = request.data.get('username', '').strip()
        organization = request.data.get(
            'organization', 'World Resources Institute')
        if not email or not username:
            return Response({"error": "Both email and username are required"}, status=status.HTTP_400_BAD_REQUEST)
        if User.objects.filter(email=email).exists() or User.objects.filter(username=username).exists():
            return Response({"error": "User with this email or username already exists"}, status=status.HTTP_400_BAD_REQUEST)
        random_password = get_random_string(length=12)
        token = get_random_string(length=32)
        try:
            with transaction.atomic():
                user = User.objects.create_user(
                    username=username, email=email, password=random_password, is_staff=True)
                UserProfile.objects.create(
                    user=user, organization=organization, invitation_status='Pending',
                    invitation_username=username, temporary_password=random_password, invitation_token=token)
                activation_link = f"{settings.FRONTEND_URL}/signin/{token}"
                context = {
                    'username': username,
                    'admin_name': request.user.get_full_name() or request.user.username,
                    'random_password': random_password,
                    'activation_link': activation_link,
                    'current_year': datetime.now().year,
                }
                html_message = render_to_string(
                    'emails/account_activation.html', context)
                email = EmailMessage(
                    subject='Your Elevate Account Activation',
                    body=html_message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[email],
                )
                email.content_subtype = "html"
                email.send()
            return Response({"message": "Invitation sent successfully"}, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            return Response({"error": "Failed to send invitation email", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class InvitedUserProfileAPI(generics.RetrieveUpdateAPIView):
    serializer_class = InvitedUserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


class SetPasswordAPI(generics.GenericAPIView):
    serializer_class = SetPasswordSerializer
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        if request.user.is_staff:
            return Response({"error": "Admins cannot use this endpoint to set passwords"}, status=status.HTTP_403_FORBIDDEN)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = request.user
        if user.profile.invitation_status != 'Pending':
            return Response({"error": "This account is already activated"}, status=status.HTTP_400_BAD_REQUEST)
        user.set_password(serializer.validated_data['new_password'])
        user.profile.invitation_status = 'Accepted'
        user.profile.temporary_password = ''
        user.profile.invitation_token = ''
        user.profile.save()
        user.save()
        return Response({"message": "Password set successfully"}, status=status.HTTP_200_OK)


class PasswordResetRequestAPI(APIView):
    permission_classes = [permissions.IsAdminUser]  # Restricted to admins

    def post(self, request):
        serializer = PasswordResetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data['email']
        try:
            user = User.objects.get(email=email)
            if not user.is_staff:
                return Response({"error": "Password reset is only allowed for admin users"}, status=status.HTTP_403_FORBIDDEN)
            token = get_random_string(length=32)
            user.profile.password_reset_token = token  # Use new field
            user.profile.save()
            frontend_url = getattr(
                settings, 'FRONTEND_URL', 'http://localhost:5173')
            reset_link = f"{frontend_url}/password-reset/{urlsafe_base64_encode(force_bytes(user.pk))}/{token}/"
            context = {
                'reset_link': reset_link,
                'current_year': datetime.now().year,
            }
            html_message = render_to_string(
                'emails/password_reset.html', context)
            email = EmailMessage(
                subject='Elevate Password Reset',
                body=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[email],
            )
            email.content_subtype = "html"
            email.send()
            return Response({"message": "Password reset email sent"}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({"error": "User with this email does not exist"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Password reset email failed: {str(e)}")
            return Response({"error": "Failed to send password reset email"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PasswordResetAPI(generics.GenericAPIView):
    serializer_class = PasswordResetNoEmailSerializer
    permission_classes = [permissions.IsAdminUser]  # Restricted to admins

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        uidb64 = serializer.validated_data['uidb64']
        token = serializer.validated_data['token']
        new_password = serializer.validated_data['new_password']
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
            if not user.is_staff:
                return Response({"error": "Password reset is only allowed for admin users"}, status=status.HTTP_403_FORBIDDEN)
            if user.profile.password_reset_token == token:  # Use new field
                user.set_password(new_password)
                user.profile.password_reset_token = ''
                user.profile.save()
                user.save()
                return Response({"message": "Password reset successfully"}, status=status.HTTP_200_OK)
            return Response({"error": "Invalid token"}, status=status.HTTP_400_BAD_REQUEST)
        except (User.DoesNotExist, ValueError):
            return Response({"error": "Invalid user or token"}, status=status.HTTP_400_BAD_REQUEST)


def password_reset_confirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
        if not user.is_staff:
            return redirect(f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')}/password-reset/invalid/")
        if user.profile.password_reset_token == token:  # Use new field
            return redirect(f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')}/password-reset/{uidb64}/{token}/")
        return redirect(f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')}/password-reset/invalid/")
    except (User.DoesNotExist, ValueError):
        return redirect(f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')}/password-reset/invalid/")


class PasswordResetLinkValidateAPI(APIView):
    permission_classes = [permissions.IsAdminUser]  # Restricted to admins

    def get(self, request, uidb64, token):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
            if not user.is_staff:
                return Response({"error": "Password reset is only allowed for admin users"}, status=status.HTTP_403_FORBIDDEN)
            if user.profile.password_reset_token == token:  # Use new field
                return Response({"message": "Valid reset link"}, status=status.HTTP_200_OK)
            return Response({"error": "Invalid reset link"}, status=status.HTTP_400_BAD_REQUEST)
        except (User.DoesNotExist, ValueError):
            return Response({"error": "Invalid reset link"}, status=status.HTTP_400_BAD_REQUEST)


class GetInvitationDetailsAPI(APIView):
    def get(self, request, token):
        try:
            profile = UserProfile.objects.get(invitation_token=token)
            return Response({
                'username': profile.invitation_username,
                'email': profile.user.email,
                'temporary_password': profile.temporary_password,
                'organization': profile.organization,
            }, status=status.HTTP_200_OK)
        except UserProfile.DoesNotExist:
            return Response({"error": "Invalid invitation token"}, status=status.HTTP_404_NOT_FOUND)


class ListInvitedUsersAPI(generics.ListAPIView):
    serializer_class = InvitedUserProfileSerializer
    permission_classes = [permissions.IsAdminUser]

    def get_queryset(self):
        return User.objects.filter(profile__invitation_status__in=['Pending', 'Accepted'])


class DeleteInvitedUserAPI(generics.DestroyAPIView):
    permission_classes = [permissions.IsAdminUser]
    queryset = User.objects.all()
    lookup_field = 'id'

    def perform_destroy(self, instance):
        instance.profile.delete()
        instance.delete()
 
        
class AnalysisView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def post(self, request, *args, **kwargs):
        logger.info(f"POST data: {request.data}")
        logger.info(f"FILES: {request.FILES}")
        created_load_categories = []
        created_vehicle_categories = []
        file_path = None
        start_time = time.time()
        try:
            logger.info(f"Starting EV analysis for user {request.user.username} at {start_time}")
            all_data = request.data.copy()
            files = request.FILES

            # Handle uploaded file (isLoadSplitFile)
            if 'isLoadSplitFile' in all_data and isinstance(all_data['isLoadSplitFile'], str):
                file_relative_path = all_data['isLoadSplitFile'].replace('/media/', '')
                file_path = os.path.join(settings.MEDIA_ROOT, file_relative_path)
                if not os.path.exists(file_path):
                    raise ValueError(f"File not found at path: {file_path}")
                all_data['isLoadSplitFile'] = file_path
            elif 'isLoadSplitFile' in files:
                file = files.get('isLoadSplitFile')
                if not file:
                    raise ValueError("No file provided for isLoadSplitFile")
                if not file.name.endswith('.xlsx'):
                    raise ValueError("Only Excel files are supported")
                fs = FileSystemStorage()
                upload_folder = 'FileUpload'
                os.makedirs(os.path.join(settings.MEDIA_ROOT, upload_folder), exist_ok=True)
                filename = fs.save(os.path.join(upload_folder, file.name), file)
                file_path = fs.path(filename)
                all_data['isLoadSplitFile'] = file_path

            # Set default name for invited users if not provided
            if 'name' not in all_data:
                all_data['name'] = request.user.username
                logger.info(f"Set default name to {request.user.username} for invited user")

            # Process categories
            processed_data, created_load_categories, created_vehicle_categories = self.process_categories(all_data)
            all_data.update(processed_data)
            all_data['user_name'] = request.user.username

            # Serialize and save instance
            serializer = AnalysisSerializer(data=all_data, context={'request': request})
            if not serializer.is_valid():
                logger.error(f"Serializer errors: {serializer.errors}")
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            ev_instance = serializer.save(user=request.user)

            # Log analysis start
            user_analysis_log = UserAnalysis.objects.create(
                userName=request.user.username,
                status='Processing',
                errorLog='',
                time=0.0
            )

            # Prepare and run analysis
            analysis_data = self.prepare_analysis_data(ev_instance)
            results = self.run_full_analysis(analysis_data, str(ev_instance.id))

            # Update log and cleanup
            user_analysis_log.status = 'Completed'
            user_analysis_log.time = time.time() - start_time
            user_analysis_log.save()

            return Response(results, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
            UserAnalysis.objects.create(
                userName=request.user.username,
                status='Failed',
                errorLog=f"{str(e)}\n{traceback.format_exc()}",
                time=time.time() - start_time
            )
            self.cleanup_temporary_data(created_load_categories, created_vehicle_categories, ev_instance, file_path)
            return Response({'error': str(e), 'traceback': traceback.format_exc()}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            self.cleanup_temporary_data(created_load_categories, created_vehicle_categories, None, file_path)

    def process_categories(self, all_data):
        processed_data = {}
        created_load_categories = []
        created_vehicle_categories = []
        load_categories = all_data.get('category_data', []) or all_data.get('categoryData', [])
        if len(load_categories) > 6:
            raise ValueError("Maximum 6 load categories allowed")
        for idx, category_data in enumerate(load_categories[:6], start=1):
            serializer = LoadCategoryModelSerializer(data=category_data)
            try:
                serializer.is_valid(raise_exception=True)
            except Exception as e:
                logger.error(f"Load category {idx} ({category_data.get('category', 'unknown')}) validation failed: {str(e)}")
                raise ValueError(f"Load category {idx} ({category_data.get('category', 'unknown')}) validation failed: {str(e)}")
            category = serializer.save()
            created_load_categories.append(category)
            processed_data[f'loadCategory{idx}'] = category.id
        vehicle_categories = all_data.get('vehicle_category_data', []) or all_data.get('vehicleCategoryData', [])
        if len(vehicle_categories) > 5:
            raise ValueError("Maximum 5 vehicle categories allowed")
        for idx, vehicle_data in enumerate(vehicle_categories[:5], start=1):
            serializer = VehicleCategoryModelSerializer(data=vehicle_data)
            try:
                serializer.is_valid(raise_exception=True)
            except Exception as e:
                logger.error(f"Vehicle category {idx} ({vehicle_data.get('vehicleCategory', 'unknown')}) validation failed: {str(e)}")
                raise ValueError(f"Vehicle category {idx} ({vehicle_data.get('vehicleCategory', 'unknown')}) validation failed: {str(e)}")
            vehicle = serializer.save()
            created_vehicle_categories.append(vehicle)
            processed_data[f'vehicleCategoryData{idx}'] = vehicle.id
        processed_data['loadCategory'] = len(load_categories)
        processed_data['numOfvehicleCategory'] = len(vehicle_categories)
        return processed_data, created_load_categories, created_vehicle_categories

    def cleanup_temporary_data(self, created_load_categories, created_vehicle_categories, ev_instance=None, file_path=None):
        with transaction.atomic():
            try:
                if created_load_categories:
                    LoadCategoryModel.objects.filter(id__in=[cat.id for cat in created_load_categories]).delete()
                    logger.info(f"Deleted {len(created_load_categories)} load categories")
            except Exception as e:
                logger.error(f"Error deleting load categories: {e}")
            try:
                if created_vehicle_categories:
                    VehicleCategoryModel.objects.filter(id__in=[veh.id for veh in created_vehicle_categories]).delete()
                    logger.info(f"Deleted {len(created_vehicle_categories)} vehicle categories")
            except Exception as e:
                logger.error(f"Error deleting vehicle categories: {e}")
            try:
                if ev_instance:
                    ev_instance.delete()
                    logger.info(f"Deleted EV analysis instance {ev_instance.id}")
            except Exception as e:
                logger.error(f"Error deleting EV analysis instance: {e}")
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

    def prepare_analysis_data(self, ev_instance):
        load_categories = []
        for i in range(1, ev_instance.loadCategory + 1):
            cat = getattr(ev_instance, f'loadCategory{i}')
            if not cat:
                raise ValueError(f"Missing loadCategory{i}")
            load_categories.append({
                'category': cat.category,
                'specifySplit': cat.specifySplit,
                'salesCAGR': cat.salesCAGR
            })
        vehicle_categories = []
        for i in range(1, ev_instance.numOfvehicleCategory + 1):
            vehicle = getattr(ev_instance, f'vehicleCategoryData{i}')
            if not vehicle:
                raise ValueError(f"Missing vehicleCategoryData{i}")
            vehicle_categories.append({
                'n': vehicle.n,
                'f': vehicle.f,
                'c': vehicle.c,
                'p': vehicle.p,
                'e': vehicle.e,
                'r': vehicle.r,
                'k': vehicle.k,
                'l': vehicle.l,
                'g': vehicle.g,
                'h': vehicle.h,
                's': vehicle.s,
                'u': vehicle.u,
                'rowlimit_xl': vehicle.rowlimit_xl,
                'CAGR_V': vehicle.CAGR_V,
                'tariff': vehicle.baseElectricityTariff
            })
        analysis_data = {
            'resolution': ev_instance.resolution,
            'BR_F': ev_instance.BR_F,
            'sharedSavaing': ev_instance.shared_saving,  # Corrected typo if model uses shared_saving
            'sum_pk_cost': ev_instance.sum_pk_cost,
            'sum_zero_cost': ev_instance.sum_zero_cost,
            'sum_op_cost': ev_instance.sum_op_cost,
            'win_pk_cost': ev_instance.win_pk_cost,
            'win_zero_cost': ev_instance.win_zero_cost,
            'win_op_cost': ev_instance.win_op_cost,
            's_pks': ev_instance.s_pks,
            's_pke': ev_instance.s_pke,
            's_sx': ev_instance.s_sx,
            's_ops': ev_instance.s_ops,
            's_ope': ev_instance.s_ope,
            's_rb': ev_instance.s_rb,
            'w_pks': ev_instance.w_pks,
            'w_pke': ev_instance.w_pke,
            'w_sx': ev_instance.w_sx,
            'w_ops': ev_instance.w_ops,
            'w_ope': ev_instance.w_ope,
            'w_rb': ev_instance.w_rb,
            'isLoadSplit': ev_instance.isLoadSplit,
            'isLoadSplitFile': ev_instance.isLoadSplitFile or None,
            'date1_start': ev_instance.date1_start,
            'date1_end': ev_instance.date1_end,
            'date2_start': ev_instance.date2_start,
            'date2_end': ev_instance.date2_end,
            'categoryData': load_categories,
            'vehicleCategoryData': vehicle_categories,
            'TOD': [
                {
                    'pks': ev_instance.s_pks,
                    'pke': ev_instance.s_pke,
                    'sx': ev_instance.s_sx,
                    'ops': ev_instance.s_ops,
                    'ope': ev_instance.s_ope,
                    'rb': ev_instance.s_rb
                },
                {
                    'pks': ev_instance.w_pks,
                    'pke': ev_instance.w_pke,
                    'sx': ev_instance.w_sx,
                    'ops': ev_instance.w_ops,
                    'ope': ev_instance.w_ope,
                    'rb': ev_instance.w_rb
                }
            ]
        }
        return analysis_data

    def run_full_analysis(self, input_data, folder_id):
        def load_forecast(n, f, c, p, e, r, k, l, g, h, s, u, rowlimit_xl, CAGR_V, tariff):
            if input_data['resolution'] <= 0:
                raise ValueError("Resolution must be positive")
            if p <= 0:
                raise ValueError("Power (p) must be positive")
            if e <= 0:
                raise ValueError("Efficiency (e) must be positive")
            Total_Charges = n * f
            Blocks = np.arange(1, int(1440/input_data['resolution'])+1, 1).reshape((1, int(1440/input_data['resolution'])))
            ex1 = np.arange(Blocks.min(), Blocks.max()+1, 1)
            mu = math.ceil(k/input_data['resolution'])
            sigma = max(math.ceil(l/input_data['resolution']), 1)
            Block_Charges = Total_Charges * ss.norm.pdf(ex1, mu, sigma)
            Block_Charges_Column = np.reshape(Block_Charges, (Blocks.max(), 1))
            Kilometers = np.arange(0, r+1, 1).reshape((1, r+1))
            StartingSOC = 100 * (1 - (Kilometers/r))
            ex2 = np.arange(0, r+1, 1).reshape(1, r+1)
            mu2 = g
            sigma2 = max(h, 1)
            Prev_Distance_Prob = ss.norm.pdf(ex2, mu2, sigma2)
            ATD = np.dot(Block_Charges_Column, Prev_Distance_Prob)
            EndingSOC = np.arange(0, 101, 1).reshape(1, 101)
            mu3 = s
            sigma3 = max(u, 1)
            EndingSOC_Prob = ss.norm.pdf(EndingSOC, mu3, sigma3)
            dummy = np.tile(StartingSOC, (101, 1))
            dummy_transpose = dummy.transpose()
            StartingSOC_Matrix = np.tile(dummy_transpose, (int(1440/input_data['resolution']), 1))
            EndingSOC_Matrix = np.tile(EndingSOC, ((int(1440/input_data['resolution']))*(r+1), 1))
            EndingSOC_Prob_Matrix = np.tile(EndingSOC_Prob, ((int(1440/input_data['resolution']))*(r+1), 1))
            ATD_Column = np.reshape(ATD, ((int(1440/input_data['resolution']))*(r+1), 1))
            Veh_All_Comb = ATD_Column * EndingSOC_Prob_Matrix
            Charging_Duration = ((60*c/input_data['resolution'])/(p*e/100)) * (EndingSOC_Matrix-StartingSOC_Matrix)
            Charging_Duration_P = np.where(Charging_Duration < 0, 0, Charging_Duration)
            Output = np.sum(Veh_All_Comb, axis=1)
            Blo_sum_linear = np.zeros(int(1440/input_data['resolution']))
            for i, value in enumerate(Output):
                block_idx = i % int(1440/input_data['resolution'])
                Blo_sum_linear[block_idx] += value
            Blo_load_sec = (p * Blo_sum_linear).reshape(1, int(1440/input_data['resolution']))
            return Blo_load_sec.tolist()

        dracula = []
        for vehicle in input_data['vehicleCategoryData']:
            try:
                res = load_forecast(**vehicle)
                dracula.append(res)
            except Exception as e:
                logger.error(f"Load forecast failed for vehicle {vehicle.get('vehicleCategory', 'unknown')}: {str(e)}")
                raise
        ddf = pd.DataFrame(np.concatenate(dracula))
        new_dict = {
            1: {'com': 0, 'ind': 0, 'res': 0, 'pub': 0, 'agr': 0, 'other': 0},
            2: {'com': 0, 'ind': 0, 'res': 0, 'pub': 0, 'agr': 0, 'other': 0}
        }
        for category in input_data['categoryData']:
            cat = category['category'][0:3] if category['category'] != "others" else category['category'][0:5]
            new_dict[1][cat] = category['specifySplit']
            new_dict[2][cat] = category['salesCAGR']
        try:
            excelimport = pd.read_excel(input_data['isLoadSplitFile'], header=None)
            if excelimport.shape[0] < 5:
                raise ValueError("Excel file must have at least 5 rows")
            if excelimport.shape[1] != 11:
                raise ValueError("Excel file must have exactly 11 columns")
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {str(e)}")
        Source = excelimport.iloc[4:, :].copy()
        Source.columns = ['Meter.No', 'datetime_utc', 'Active_B_PH', 'Active_Y_PH', 'Active_R_PH', 'Reactive_B_PH', 'Reactive_Y_PH', 'Reactive_R_PH', 'VBV', 'VYV', 'VRV']
        Source = Source.reset_index(drop=True)
        try:
            Source[''] = ((Source.Active_B_PH * Source.VBV) + (Source.Active_Y_PH * Source.VYV) + (Source.Active_R_PH * Source.VRV))/1000
            Source['datetime_utc'] = pd.to_datetime(Source['datetime_utc'], errors='coerce')
            if Source['datetime_utc'].isna().any():
                raise ValueError("Invalid datetime values in Excel file")
            Source['date'] = Source['datetime_utc'].dt.date
        except Exception as e:
            raise ValueError(f"Error processing Excel data: {str(e)}")
        labels = Source['datetime_utc'].dt.date.unique()
        slots = pd.DataFrame(Source['datetime_utc'].dt.time.unique())
        slots.columns = ['slot_labels']
        value_to_repeat = excelimport.iloc[0, 1] * (float(input_data['BR_F'])/100)
        number_of_repeats = int(1440/input_data['resolution'])
        Transformer_capacity = pd.DataFrame(np.repeat(value_to_repeat, number_of_repeats))
        Transformer_capacity.columns = ['Transformer safety planning trigger']
        Transformer_capacity['100% Transformer rated capacity'] = np.repeat(excelimport.iloc[0, 1], number_of_repeats)
        Source.set_index('datetime_utc', inplace=True)
        Calculated_load = Source.drop(['Meter.No', 'Active_B_PH', 'Active_Y_PH', 'Active_R_PH', 'Reactive_B_PH', 'Reactive_Y_PH', 'Reactive_R_PH', 'VBV', 'VYV', 'VRV'], axis=1)
        Calculated_load.index.name = None
        abc = pd.DataFrame(Calculated_load[''].astype(float).values)
        resolution = input_data['resolution']
        time_blocks_per_day = int(1440 / resolution)
        total_data_points = len(abc)
        complete_days = total_data_points // time_blocks_per_day
        if total_data_points % time_blocks_per_day != 0:
            abc_trimmed = abc.iloc[:complete_days * time_blocks_per_day]
            logger.warning(f"Data trimmed from {total_data_points} to {len(abc_trimmed)} points to fit {complete_days} complete days")
        else:
            abc_trimmed = abc
        try:
            load_extract = pd.DataFrame(np.reshape(abc_trimmed.to_numpy(), (complete_days, time_blocks_per_day)))
            load_extract = load_extract.T
        except ValueError as e:
            logger.error(f"Reshape error: {e}")
            raise ValueError(f"Data reshape failed: {str(e)}")
        if len(labels) != load_extract.shape[1]:
            if len(labels) > load_extract.shape[1]:
                labels = labels[:load_extract.shape[1]]
                logger.warning(f"Trimmed labels to {len(labels)} to match data columns")
            else:
                additional_needed = load_extract.shape[1] - len(labels)
                labels_to_repeat = labels[-additional_needed:] if additional_needed <= len(labels) else labels
                labels = np.concatenate([labels, labels_to_repeat[:additional_needed]])
                logger.warning(f"Extended labels to {len(labels)} to match data columns")
        final_load = load_extract.copy()
        final_load.columns = labels
        max_cols = min(406, final_load.shape[1])
        selected_range = final_load.iloc[:, :max_cols].copy()
        selected_ranges = [selected_range]
        for year in range(1, 5):
            growth_factors = {cat: new_dict[1][cat] * (1 + new_dict[2][cat] / 100) for cat in new_dict[1]}
            next_range = (selected_ranges[-1]/100) * sum(growth_factors.values())
            selected_ranges.append(next_range)
        selected_range_2, selected_range_3, selected_range_4, selected_range_5 = selected_ranges[1:5]
        shazam = pd.DataFrame(ddf.sum(axis=0))
        growth_factors = [(input_data['vehicleCategoryData'][i]['CAGR_V'])/100 + 1 for i in range(len(input_data['vehicleCategoryData']))]
        ddf2 = ddf.mul(growth_factors, axis=0)
        shazam_2 = pd.DataFrame(ddf2.sum(axis=0).to_numpy())
        ddf3 = ddf2.mul(growth_factors, axis=0)
        shazam_3 = pd.DataFrame(ddf3.sum(axis=0).to_numpy())
        ddf4 = ddf3.mul(growth_factors, axis=0)
        shazam_4 = pd.DataFrame(ddf4.sum(axis=0).to_numpy())
        ddf5 = ddf4.mul(growth_factors, axis=0)
        shazam_5 = pd.DataFrame(ddf5.sum(axis=0).to_numpy())

        def TOD_key(pks, pke, sx, ops, ope, rb):
            try:
                pke = pke.replace("24:00", "23:59") if pke else "23:59"
                ope = ope.replace("24:00", "23:59") if ope else "23:59"
                FMT = '%H:%M'
                pks = datetime.strptime(str(pks or "00:00"), FMT).time()
                pke = datetime.strptime(str(pke or "23:59"), FMT).time()
                ops = datetime.strptime(str(ops or "00:00"), FMT).time()
                ope = datetime.strptime(str(ope or "23:59"), FMT).time()
                TOD_Duration = {
                    'peak_hours': (datetime.combine(datetime.today(), pke) - datetime.combine(datetime.today(), pks)).seconds / 3600,
                    'offpeak_hours': (datetime.combine(datetime.today(), ope) - datetime.combine(datetime.today(), ops)).seconds / 3600,
                }
                return TOD_Duration
            except ValueError as e:
                logger.error(f"Invalid time format in TOD_key: {str(e)}")
                raise ValueError(f"Invalid time format: {str(e)}")

        TOD_m = []
        for tod_data in input_data['TOD']:
            res = TOD_key(**tod_data)
            TOD_m.append(res)
        TOD_matrix = pd.DataFrame(TOD_m)
        output_data = {}
        output_data['Simulated_EV_Load'] = ddf.values.tolist()
        ddf_years = [ddf]
        shazam_years = [shazam]
        ev_loads = []
        for year in range(1, 6):
            last_ddf = ddf_years[-1]
            ddf_next = last_ddf.mul(growth_factors, axis=0)
            ddf_years.append(ddf_next)
            shazam = pd.DataFrame(ddf_next.sum(axis=0).to_numpy())
            shazam_years.append(shazam)
            ev_loads.append({f'Year {year}': ddf_next.sum(axis=0).to_numpy().flatten().tolist()})
        output_data['EV_Load'] = ev_loads
        try:
            start_date = "2019-05-01"
            end_date = "2020-07-30"
            dates = pd.date_range(start=start_date, end=end_date)
        except ValueError as e:
            logger.error(f"Invalid date range: {str(e)}")
            raise ValueError(f"Invalid date range: {str(e)}")
        time_slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
        pivot_df = pd.DataFrame(
            index=time_slots, columns=dates.date, dtype=float)
        for i, date in enumerate(dates):
            base_pattern = np.array([60 + 40 * np.sin(2 * np.pi * (h) / 24) + np.random.normal(0, 5) for h in np.linspace(0, 24, 48)])
            if date.weekday() < 5:
                base_pattern *= 1.0 + 0.1 * np.random.random()
            else:
                base_pattern *= 0.8 + 0.1 * np.random.random()
            month = date.month
            if month in [6, 7, 8]:
                base_pattern *= 1.1 + 0.05 * np.random.random()
            elif month in [12, 1, 2]:
                base_pattern *= 1.05 + 0.05 * np.random.random()
            cost_weight = np.tile(np.random.rand(5), 10)[:48] * 100
            daily_load = 0.7 * base_pattern + 0.3 * cost_weight
            daily_load += np.random.normal(0, 3, size=len(daily_load))
            pivot_df[date.date()] = np.round(daily_load.astype(float), 6)
        output_data['DT_Base_Load'] = pivot_df.values.tolist()
        selected_ranges = [selected_range, selected_range_2, selected_range_3, selected_range_4, selected_range_5]
        base_loads = []
        for year in range(5):
            if year >= len(selected_ranges):
                continue
            base_load = selected_ranges[year]
            if base_load is None or base_load.empty:
                continue
            mean_load = base_load.mean(axis=1).values.tolist()
            base_loads.append({f'Year {year+1}': mean_load})
        output_data['Base_Load'] = base_loads
        required_net_loads = [pd.DataFrame(selected_ranges[year]) + shazam_years[year].values for year in range(5)]
        combined_loads = []
        for year in range(5):
            base_load = pd.DataFrame(selected_ranges[year]).mean(axis=1).values.tolist()
            ev_load = shazam_years[year].mean(axis=1).values.tolist()
            combined_loads.append({f'Year {year+1}': {'base_load': base_load, 'ev_load': ev_load}})
        output_data['Base_EV_Load'] = combined_loads
        final_res_res = pd.DataFrame(np.random.rand(5, int(1440/input_data['resolution'])))
        output_data['Base_ToD_EV_Load'] = self.generate_tod_ev_load_plot(final_res_res, excelimport, input_data['resolution'])
        overshots = [rn - (excelimport.iloc[0, 1] * (float(input_data['BR_F'])/100)) for rn in required_net_loads]
        overshots_r = [rn - (excelimport.iloc[0, 1] * (90/100)) for rn in required_net_loads]
        table_data = []
        for year in range(5):
            table_data.append({
                'Year': f'Year {year+1}',
                'Max_excursion_planning': round(overshots[year].max().max(), 2) if not overshots[year].empty else 0,
                'Num_excursions_planning': (overshots[year] > 0).values.sum() if not overshots[year].empty else 0,
                'Max_excursion_rated': round(overshots_r[year].max().max(), 2) if not overshots_r[year].empty else 0,
                'Num_excursions_rated': (overshots_r[year] > 0).values.sum() if not overshots_r[year].empty else 0
            })
        output_data['Summary_Table'] = table_data
        overshot_data = []
        for year in range(1, 6):
            ov = overshots[year-1]
            ov_r = overshots_r[year-1]
            ov_p = ov.reset_index().melt(id_vars=['index']).rename(columns={
                'index': 'slot',
                'variable': 'dates',
                'value': f'overshot_planning_year_{year}'
            })
            ov_r_p = ov_r.reset_index().melt(id_vars=['index']).rename(columns={
                'index': 'slot',
                'variable': 'dates',
                'value': f'overshot_rated_year_{year}'
            })
            overshot_data.append({f'Year {year}': {'planning': ov_p.to_dict('records'), 'rated': ov_r_p.to_dict('records')}})
        output_data['Overshot'] = overshot_data
        abc_ddf = pd.DataFrame({'seasons': np.random.choice(['summer', 'winter'], 5), 'unit_type': np.random.choice(['pk', 'op', '0'], 5)})
        conditions_4 = [
            (abc_ddf['seasons'] == 'summer') & (abc_ddf['unit_type'] == 'pk'),
            (abc_ddf['seasons'] == 'winter') & (abc_ddf['unit_type'] == 'pk'),
            (abc_ddf['seasons'] == 'summer') & (abc_ddf['unit_type'] == 'op'),
            (abc_ddf['seasons'] == 'winter') & (abc_ddf['unit_type'] == 'op'),
            (abc_ddf['seasons'] == 'summer') & (abc_ddf['unit_type'] == '0'),
            (abc_ddf['seasons'] == 'winter') & (abc_ddf['unit_type'] == '0')
        ]
        values_4 = [
            input_data['sum_pk_cost'], input_data['win_pk_cost'],
            input_data['sum_op_cost'], input_data['win_op_cost'],
            input_data['sum_zero_cost'], input_data['win_zero_cost']
        ]
        abc_ddf['utility_proc_tariff'] = np.select(conditions_4, values_4)
        ddf_x = final_res_res.copy()
        ddf_x.index = abc_ddf.index
        old_utility_cost = (ddf_x.mul(abc_ddf['utility_proc_tariff'], axis=0) / 2)
        old_utility_cost.columns = ddf_x.columns.astype(str)
        old_utility_cost = old_utility_cost.rename(columns=lambda x: x + '_old_cost')
        final_res = pd.DataFrame(np.random.rand(5, int(1440 / input_data['resolution'])))
        new_utility_cost = (final_res.mul(abc_ddf['utility_proc_tariff'], axis=0) / 2)
        new_utility_cost.columns = final_res.columns.astype(str)
        new_utility_cost = new_utility_cost.rename(columns=lambda x: x + '_new_cost')
        retail_tariff_df = pd.DataFrame({'1': [1], '2': [1], '3': [1], '4': [1]})
        retail_tariff_value = retail_tariff_df.iloc[0].mean()
        old_tariff_revenue = (ddf_x * retail_tariff_value / 2)
        old_tariff_revenue.columns = ddf_x.columns.astype(str)
        old_tariff_revenue = old_tariff_revenue.rename(columns=lambda x: x + '_old_tariff')
        abc_ddf = pd.concat([abc_ddf, old_utility_cost, new_utility_cost, old_tariff_revenue], axis=1)
        output_data['Load_Simulation_ToD_Calculation'] = abc_ddf.to_dict('records')

        def calc_TOD_x(year_num):
            try:
                s_pk_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (abc_ddf['unit_type'] == 'pk'), f'{year_num}_old_cost'].sum()
                w_pk_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (abc_ddf['unit_type'] == 'pk'), f'{year_num}_old_cost'].sum()
                s_op_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (abc_ddf['unit_type'] == 'op'), f'{year_num}_old_cost'].sum()
                w_op_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (abc_ddf['unit_type'] == 'op'), f'{year_num}_old_cost'].sum()
                s_0_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (abc_ddf['unit_type'] == '0'), f'{year_num}_old_cost'].sum()
                w_0_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (abc_ddf['unit_type'] == '0'), f'{year_num}_old_cost'].sum()
                old_tariff_sum = abc_ddf[f'{year_num}_old_tariff'].sum()
                cost_diff = abc_ddf[f'{year_num}_new_cost'].sum() - abc_ddf[f'{year_num}_old_cost'].sum()
                shared_saving = float(input_data['sharedSavaing']) / 100
                if retail_tariff_df.iloc[0, 0] == 0:
                    raise ValueError("Retail tariff cannot be zero")
                numerator = (60 / input_data['resolution'] * (old_tariff_sum + cost_diff) * shared_saving) / retail_tariff_df.iloc[0, 0]
                denominator = (s_pk_sum + w_pk_sum - s_op_sum - w_op_sum)
                if denominator == 0:
                    logger.warning(f"Zero denominator in calc_TOD_x for year {year_num}, returning 0")
                    return 0
                result = 100 * (numerator - (s_pk_sum + w_pk_sum + s_op_sum + w_op_sum + s_0_sum + w_0_sum)) / denominator
                return round(result, 2)
            except Exception as e:
                logger.error(f"Error in calc_TOD_x for year {year_num}: {str(e)}")
                return 0
        output_data['TOD_Surcharge_Rebate'] = [calc_TOD_x(i) for i in range(1, 5)]
        return output_data

    def generate_tod_ev_load_plot(self, final_res_res, excelimport, resolution):
        if final_res_res.empty:
            raise ValueError("Input data for ToD plot is empty")
        final_res_res_1 = final_res_res.to_numpy()
        mean_vals = final_res_res_1.mean(axis=0).tolist()
        std_vals = final_res_res_1.std(axis=0).tolist()
        time_blocks = np.arange(int(1440/resolution)).tolist()
        return {'time_blocks': time_blocks, 'mean_load': mean_vals, 'std_dev': std_vals}


class AnalysisListCreateAPI(generics.ListCreateAPIView):
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
    queryset = Analysis.objects.all()
    serializer_class = AnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]
    lookup_field = 'id'

    def get_queryset(self):
        if self.request.user.is_staff:
            return Analysis.objects.all()
        return Analysis.objects.filter(user=self.request.user)

    def perform_update(self, serializer):
        serializer.save(user=self.request.user)


class FileUploadAPI(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        file_serializer = FilesSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DeleteDataAPI(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        folder_id = request.data.get('folderId')
        try:
            is_folder = os.path.isdir(os.path.join(
                settings.MEDIA_ROOT, 'outputs', str(folder_id)))
            if is_folder:
                shutil.rmtree(os.path.join(settings.MEDIA_ROOT,
                              'outputs', str(folder_id)), ignore_errors=True)
            analysis = Analysis.objects.get(
                id=folder_id, user=self.request.user)
            for load_cat in [analysis.loadCategory1, analysis.loadCategory2, analysis.loadCategory3, analysis.loadCategory4, analysis.loadCategory5, analysis.loadCategory6]:
                if load_cat:
                    load_cat.delete()
            for vehi_cat in [analysis.vehicleCategoryData1, analysis.vehicleCategoryData2, analysis.vehicleCategoryData3, analysis.vehicleCategoryData4, analysis.vehicleCategoryData5]:
                if vehi_cat:
                    vehi_cat.delete()
            analysis.delete()
            return Response({"message": "Data deleted successfully"}, status=status.HTTP_200_OK)
        except Analysis.DoesNotExist:
            return Response({"message": "Analysis not found, folder deleted if existed"}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error deleting analysis: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserAnalysisLogAPI(generics.ListAPIView):
    serializer_class = UserAnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if self.request.user.is_staff:
            return UserAnalysis.objects.all()
        return UserAnalysis.objects.filter(userName=self.request.user.username)


class IsAdminUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user and request.user.is_staff


class PermanentAnalysisListCreateAPI(generics.ListCreateAPIView):
    queryset = PermanentAnalysis.objects.all()
    serializer_class = PermanentAnalysisSerializer
    permission_classes = [IsAdminUser]

    def get_serializer_context(self):
        return {'request': self.request}



class PermanentAnalysisRetrieveUpdateDestroyAPI(generics.RetrieveUpdateDestroyAPIView):
    queryset = PermanentAnalysis.objects.all()
    serializer_class = PermanentAnalysisSerializer
    permission_classes = [IsAdminUser]
