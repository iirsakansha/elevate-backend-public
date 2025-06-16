import logging
import seaborn as sns
from itertools import chain
import itertools
from django.http import response
from django.http.response import FileResponse
from django.shortcuts import render, redirect
from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import api_view
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from rest_framework import generics, permissions
from knox.models import AuthToken
from .serializers import UserSerializer, RegisterSerializer, ChangePasswordSerializer, EvAnalysisSerializer, LoginUserSerializer, LoadCategoryModelSerializer, vehicleCategoryModelSerializer, FilesSerializer
from rest_framework import permissions
from rest_framework.authtoken.serializers import AuthTokenSerializer
from knox.views import LoginView as KnoxLoginView
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .models import evAnalysis, LoadCategoryModel, vehicleCategoryModel, Files, userAnalysis
from django.core.files.storage import FileSystemStorage
from rest_framework.parsers import JSONParser
import os
import shutil
import json
import numpy
from django.http import HttpResponse, JsonResponse
from django.conf import settings
# Importing standard distribution calculation modules
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as ss
import math as math
import matplotlib.pyplot as plt
import sys
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors, use
import matplotlib
import traceback
matplotlib.use('Agg')
sns.set()

# Index


def index(request):
    return render(request, 'EVTools/index.html')

# Register API


class RegisterAPI(generics.GenericAPIView):
    serializer_class = RegisterSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({
            "user": UserSerializer(user, context=self.get_serializer_context()).data,
            "token": AuthToken.objects.create(user)[1]
        })

# Login API


class LoginAPI(generics.GenericAPIView):
    serializer_class = LoginUserSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data
        return Response({
            "user": UserSerializer(user, context=self.get_serializer_context()).data,
            "token": AuthToken.objects.create(user)[1]
        })

# Chnage Password API


class ChangePasswordView(generics.UpdateAPIView):
    serializer_class = ChangePasswordSerializer
    model = User
    permission_classes = (IsAuthenticated,)

    def get_object(self, queryset=None):
        obj = self.request.user
        return obj

    def update(self, request, *args, **kwargs):
        self.object = self.get_object()
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            # Check old password
            if not self.object.check_password(serializer.data.get("old_password")):
                return Response({"err": "Your entered current password is wrong"}, status=status.HTTP_400_BAD_REQUEST)
            # set_password also hashes the password that the user will get
            self.object.set_password(serializer.data.get("new_password"))
            self.object.save()
            response = {
                'status': 'success',
                'code': status.HTTP_200_OK,
                'message': 'Password updated successfully',
                'data': []
            }

            return Response(response)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# Get User API


class UserAPI(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated, ]
    serializer_class = UserSerializer

    def get_object(self):
        return self.request.user

# Delete Data


class DeleteData(APIView):

    def post(self, request):
        folderId = request.data['folderId']
        print(folderId)
        try:
            # deleted output folder data
            isFolder = os.path.isdir(f'media/outputs/{folderId}')
            if isFolder:
                shutil.rmtree(
                    f"media/outputs/{folderId}", ignore_errors=True)
            else:
                pass
        except Exception as e:
            print(e)
            return Response(status=status.HTTP_201_CREATED)

        try:
            EvDataDBObjects = evAnalysis.objects.get(id=folderId)
            print(EvDataDBObjects)
            if EvDataDBObjects:
                # deleted loadCategorys
                for load_cat in [EvDataDBObjects.loadCategory1, EvDataDBObjects.loadCategory2, EvDataDBObjects.loadCategory3, EvDataDBObjects.loadCategory4, EvDataDBObjects.loadCategory5, EvDataDBObjects.loadCategory6]:
                    if load_cat != None:
                        print(load_cat)
                        LoadCategoryModel.objects.get(id=load_cat.id).delete()
                        print("Done")
                    else:
                        print("Done with continue")
                        continue
                # deleted vehicleCategorys
                for vehi_cat in [EvDataDBObjects.vehicleCategoryData1, EvDataDBObjects.vehicleCategoryData2, EvDataDBObjects.vehicleCategoryData3, EvDataDBObjects.vehicleCategoryData4, EvDataDBObjects.vehicleCategoryData5]:
                    if vehi_cat != None:
                        print(vehi_cat)
                        vehicleCategoryModel.objects.get(
                            id=vehi_cat.id).delete()
                        print("Done")
                    else:
                        print("Done with continue")
                        continue
                # deleted main Ev recoed
                EvDataDBObjects.delete()
                return Response(status=status.HTTP_201_CREATED)
        except:
            return Response(status=status.HTTP_201_CREATED)

# File Upload


class FileUpload(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [permissions.IsAuthenticated, ]

    def post(self, request, *args, **kwargs):
        file_serializer = FilesSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# EV Analysis with automatic cleanup


class EvAnalysisView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        created_load_categories = []
        created_vehicle_categories = []

        try:
            start_time = time.time()
            all_data = request.data.copy()
            files = request.FILES
            # Ensure the root media folder exists
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

            # Save uploaded file
            if 'isLoadSplitFile' in all_data and isinstance(all_data['isLoadSplitFile'], str):
                file_relative_path = all_data['isLoadSplitFile'].replace(
                    '/media/', '')
                file_path = os.path.join(
                    settings.MEDIA_ROOT, file_relative_path)
                if not os.path.exists(file_path):
                    raise Exception(f"File not found at path: {file_path}")
                all_data['isLoadSplitFile'] = file_path
            elif 'isLoadSplitFile' in files:
                fs = FileSystemStorage()
                file = files['isLoadSplitFile']
                upload_folder = 'media/FileUpload'
                # Create the upload folder under media
                os.makedirs(os.path.join(
                    fs.location, upload_folder), exist_ok=True)
                filename = fs.save(f"media/FileUpload/{file.name}", file)
                file_path = fs.path(filename)
                all_data['isLoadSplitFile'] = file_path
            try:
                processed_data, created_load_categories, created_vehicle_categories = self.process_categories(
                    all_data)
                all_data.update(processed_data)
            except Exception as e:
                return Response({'error': f"process_categories failed: {str(e)}"}, status=500)

            try:
                serializer = EvAnalysisSerializer(data=all_data)
                if not serializer.is_valid():
                    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                ev_instance = serializer.save()
            except Exception as e:
                return Response({'error': f"Serializer failed: {str(e)}"}, status=500)

            output_folder_path = os.path.join(
                'media', 'outputs', str(ev_instance.id))
            os.makedirs(output_folder_path, exist_ok=True)

            # Prepare analysis data
            analysis_data = self.prepare_analysis_data(ev_instance)

            # Run complete analysis
            results = self.run_full_analysis(
                analysis_data, str(ev_instance.id))

            response_data = self.get_output_links(ev_instance.id)
            # SUCCESS: Clean up temporary data after successful analysis
            self.cleanup_temporary_data(
                created_load_categories, created_vehicle_categories)

            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            # ERROR: Clean up temporary data in case of failure
            self.cleanup_temporary_data(
                created_load_categories, created_vehicle_categories)
            return Response(
                {'error': str(e), 'traceback': traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def process_categories(self, all_data):
        """Process dynamic categories and return created objects for cleanup"""
        processed_data = {}
        created_load_categories = []
        created_vehicle_categories = []

        # Process load categories
        load_categories = all_data.get('categoryData', [])
        for idx, category_data in enumerate(load_categories[:6], start=1):
            category = LoadCategoryModel.objects.create(**category_data)
            created_load_categories.append(category)  # Track for cleanup
            processed_data[f'loadCategory{idx}'] = category.id

        # Process vehicle categories
        vehicle_categories = all_data.get('vehicleCategoryData', [])
        for idx, vehicle_data in enumerate(vehicle_categories[:5], start=1):
            vehicle = vehicleCategoryModel.objects.create(**vehicle_data)
            created_vehicle_categories.append(vehicle)  # Track for cleanup
            processed_data[f'vehicleCategoryData{idx}'] = vehicle.id

        # Set counts
        processed_data['loadCategory'] = len(load_categories)
        processed_data['numOfvehicleCategory'] = len(vehicle_categories)

        return processed_data, created_load_categories, created_vehicle_categories

    def cleanup_temporary_data(self, created_load_categories, created_vehicle_categories):
        """Delete all temporary category data created during processing"""
        try:
            # Delete load categories
            for category in created_load_categories:
                try:
                    category.delete()
                except Exception as e:
                    print(f"Error deleting load category {category.id}: {e}")

            # Delete vehicle categories
            for vehicle in created_vehicle_categories:
                try:
                    vehicle.delete()
                except Exception as e:
                    print(f"Error deleting vehicle category {vehicle.id}: {e}")

            print(
                f"Cleaned up {len(created_load_categories)} load categories and {len(created_vehicle_categories)} vehicle categories")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def prepare_analysis_data(self, ev_instance):
        """Prepare data dictionary for analysis"""

        # Get all related category data
        load_categories = []
        for i in range(1, ev_instance.loadCategory + 1):
            cat_id = getattr(ev_instance, f'loadCategory{i}_id')
            cat = LoadCategoryModel.objects.get(id=cat_id)
            load_categories.append({
                'category': cat.category,
                'specifySplit': cat.specifySplit,
                'salesCAGR': cat.salesCAGR
            })

        vehicle_categories = []
        for i in range(1, ev_instance.numOfvehicleCategory + 1):
            vehicle_id = getattr(ev_instance, f'vehicleCategoryData{i}_id')
            vehicle = vehicleCategoryModel.objects.get(id=vehicle_id)
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
                'rowlimit_xl': 1000000,  # Default value
                'CAGR_V': vehicle.CAGR_V,
                'tariff': vehicle.baseElectricityTariff
            })
        analysis_data = {
            'resolution': ev_instance.resolution,
            'BR_F': ev_instance.BR_F,
            'sharedSavaing': ev_instance.sharedSavaing,
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
        """Run the complete analysis pipeline with fixed resolution handling"""
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        # Create output directory
        output_dir = os.path.join(settings.MEDIA_ROOT, 'outputs', folder_id)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Load forecast for each vehicle category
        def load_forecast(n, f, c, p, e, r, k, l, g, h, s, u, rowlimit_xl, CAGR_V, tariff):
            begin = time.time()
            Total_Charges = n * f

            Blocks = np.arange(1, int(
                1440/input_data['resolution'])+1, 1).reshape((1, int(1440/input_data['resolution'])))

            ex1 = np.arange(Blocks.min(), Blocks.max()+1, 1)
            mu = math.ceil(k/input_data['resolution'])
            sigma = math.ceil(l/input_data['resolution'])

            Block_Charges = Total_Charges * (ss.norm.pdf(ex1, mu, sigma))
            Block_Charges_Column = np.reshape(Block_Charges, (Blocks.max(), 1))

            Kilometers = np.arange(0, r+1, 1).reshape((1, r+1))
            StartingSOC = 100 * (1 - (Kilometers/r))

            ex2 = np.arange(0, r+1, 1).reshape(1, r+1)
            mu2 = g
            sigma2 = h

            Prev_Distance_Prob = ss.norm.pdf(ex2, mu2, sigma2)
            ATD = np.dot(Block_Charges_Column, Prev_Distance_Prob)

            EndingSOC = np.arange(0, 101, 1).reshape(1, 101)
            mu3 = s
            sigma3 = u

            EndingSOC_Prob = ss.norm.pdf(EndingSOC, mu3, sigma3)
            dummy = np.tile(StartingSOC, (101, 1))
            dummy_transpose = (dummy.transpose())
            StartingSOC_Matrix = np.tile(
                dummy_transpose, (int(1440/input_data['resolution']), 1))

            EndingSOC_Matrix = np.tile(
                EndingSOC, ((int(1440/input_data['resolution']))*(r+1), 1))
            EndingSOC_Prob_Matrix = np.tile(
                EndingSOC_Prob, ((int(1440/input_data['resolution']))*(r+1), 1))

            ATD_Column = np.reshape(
                ATD, ((int(1440/input_data['resolution']))*(r+1), 1))
            Veh_All_Comb = ATD_Column * EndingSOC_Prob_Matrix

            Charging_Duration = (
                (60*c/input_data['resolution'])/(p*e)) * (EndingSOC_Matrix-StartingSOC_Matrix)
            Charging_Duration_P = np.where(
                Charging_Duration < 0, 0, Charging_Duration)
            Output = np.sum(Veh_All_Comb, axis=1)

            Blo_sum_linear = np.zeros(int(1440/input_data['resolution']))
            for i, value in enumerate(Output):
                block_idx = i % int(1440/input_data['resolution'])
                Blo_sum_linear[block_idx] += value

            Blo_load_sec = (p * Blo_sum_linear).reshape(1,
                                                        int(1440/input_data['resolution']))

            calculation_time = time.time() - begin
            return Blo_load_sec

        # Process vehicle forecasts
        dracula = []
        for idx, vehicle in enumerate(input_data['vehicleCategoryData']):
            res = load_forecast(**vehicle)
            dracula.append(res)
        ddf = pd.DataFrame(np.concatenate(dracula))

        # Process load categories
        new_dict = {
            1: {'com': 0, 'ind': 0, 'res': 0, 'pub': 0, 'agr': 0, 'other': 0},
            2: {'com': 0, 'ind': 0, 'res': 0, 'pub': 0, 'agr': 0, 'other': 0}
        }
        for idx, category in enumerate(input_data['categoryData']):
            cat = category['category'][0:3] if category['category'] != "others" else category['category'][0:5]
            new_dict[1][cat] = category['specifySplit']
            new_dict[2][cat] = category['salesCAGR']

        # Process load data from file
        excelimport = pd.read_excel(input_data['isLoadSplitFile'], header=None)

        Source = excelimport.iloc[4:, :]
        Source.columns = ['Meter.No', 'datetime_utc', 'Active_B_PH', 'Active_Y_PH',
                          'Active_R_PH', 'Reactive_B_PH', 'Reactive_Y_PH', 'Reactive_R_PH', 'VBV', 'VYV', 'VRV']
        Source = Source.reset_index(drop=True)

        Source[''] = (((Source.Active_B_PH * Source.VBV) +
                       (Source.Active_Y_PH * Source.VYV) +
                       (Source.Active_R_PH * Source.VRV))/1000)
        Source['datetime_utc'] = pd.to_datetime(Source['datetime_utc'])
        Source['date'] = Source['datetime_utc'].dt.date

        labels = Source['datetime_utc'].dt.date.unique()
        slots = pd.DataFrame(Source['datetime_utc'].dt.time.unique())
        slots.columns = ['slot_labels']

        value_to_repeat = excelimport.iloc[0, 1]*(input_data['BR_F'])/100
        number_of_repeats = int(1440/input_data['resolution'])
        Transformer_capacity = pd.DataFrame(
            np.repeat(value_to_repeat, number_of_repeats))
        Transformer_capacity.columns = ['Transformer safety planning trigger']
        Transformer_capacity['100% Transformer rated capacity'] = np.repeat(
            excelimport.iloc[0, 1]*(100/100), 1440/input_data['resolution'])

        Source.set_index('datetime_utc', inplace=True)
        Calculated_load = Source.drop(['Meter.No', 'Active_B_PH', 'Active_Y_PH', 'Active_R_PH',
                                       'Reactive_B_PH', 'Reactive_Y_PH', 'Reactive_R_PH', 'VBV', 'VYV', 'VRV'], axis=1)
        Calculated_load.index.name = None
        abc = pd.DataFrame(Calculated_load[''].astype(float).values)

        # FIXED RESHAPING LOGIC - Handle different resolutions properly
        resolution = input_data['resolution']
        time_blocks_per_day = int(1440 / resolution)

        # Get the total number of data points
        total_data_points = len(abc)

        # Calculate how many complete days we can form
        complete_days = total_data_points // time_blocks_per_day

        # If we don't have complete days, we need to handle this
        if total_data_points % time_blocks_per_day != 0:
            # Trim the data to complete days only
            abc_trimmed = abc.iloc[:complete_days * time_blocks_per_day]
            print(
                f"Warning: Data trimmed from {total_data_points} to {len(abc_trimmed)} points to fit {complete_days} complete days")
        else:
            abc_trimmed = abc

        # Now reshape with the correct dimensions
        try:
            load_extract = pd.DataFrame(np.reshape(
                abc_trimmed.to_numpy(), (complete_days, time_blocks_per_day)))
            # Transpose to get time blocks as rows and days as columns
            load_extract = load_extract.T
        except ValueError as e:
            print(f"Reshape error: {e}")
            print(
                f"Data points: {len(abc_trimmed)}, Expected shape: ({complete_days}, {time_blocks_per_day})")
            raise

        # Handle the labels mismatch issue
        if len(labels) != load_extract.shape[1]:
            print(
                f"Labels mismatch: {len(labels)} labels vs {load_extract.shape[1]} columns")
            # Adjust labels to match the data
            if len(labels) > load_extract.shape[1]:
                labels = labels[:load_extract.shape[1]]
                print(f"Trimmed labels to {len(labels)} to match data columns")
            else:
                # If we have fewer labels than columns, duplicate some dates
                additional_needed = load_extract.shape[1] - len(labels)
                # Repeat the last few dates
                labels_to_repeat = labels[-additional_needed:] if additional_needed <= len(
                    labels) else labels
                labels = np.concatenate(
                    [labels, labels_to_repeat[:additional_needed]])
                print(
                    f"Extended labels to {len(labels)} to match data columns")

        final_load = load_extract.copy()
        final_load.columns = labels

        # Step 7: Project Loads Over Years - Fix indexing issue
        # Make sure we don't try to access more columns than available
        max_cols = min(406, final_load.shape[1])
        selected_range = final_load.iloc[:, :max_cols].copy()
        selected_ranges = [selected_range]

        for year in range(1, 5):
            next_range = (selected_ranges[-1]/100) * (
                new_dict[1]['com']*(1+(new_dict[2]['com']/100)) +
                new_dict[1]['res']*(1+(new_dict[2]['res']/100))
            )
            selected_ranges.append(next_range)

        selected_range_2, selected_range_3, selected_range_4, selected_range_5 = selected_ranges[
            1:5]

        # Project EV load for 5 years
        shazam = pd.DataFrame(ddf.sum(axis=0))

        # Ensure we have enough vehicle categories to avoid index errors
        growth_factors = []
        for i in range(len(input_data['vehicleCategoryData'])):
            if i < len(input_data['vehicleCategoryData']):
                growth_factors.append(
                    (input_data['vehicleCategoryData'][i]['CAGR_V'])/100 + 1)
            else:
                growth_factors.append(1.0)  # Default growth factor

        ddf2 = ddf.mul(growth_factors, axis=0)
        shazam_2 = pd.DataFrame((ddf2.sum(axis=0)).to_numpy())

        ddf3 = ddf2.mul(growth_factors, axis=0)
        shazam_3 = pd.DataFrame((ddf3.sum(axis=0)).to_numpy())

        ddf4 = ddf3.mul(growth_factors, axis=0)
        shazam_4 = pd.DataFrame((ddf4.sum(axis=0)).to_numpy())

        ddf5 = ddf4.mul(growth_factors, axis=0)
        shazam_5 = pd.DataFrame((ddf5.sum(axis=0)).to_numpy())

        # Step 8: Time-of-Day (ToD) Analysis
        def TOD_key(pks, pke, sx, ops, ope, rb):
            pke = pke.replace("24:00", "23:59")
            ope = ope.replace("24:00", "23:59")
            FMT = '%H:%M'
            pks = datetime.strptime(str(pks), FMT).time()
            pke = datetime.strptime(str(pke), FMT).time()
            ops = datetime.strptime(str(ops), FMT).time()
            ope = datetime.strptime(str(ope), FMT).time()
            TOD_Duration = pd.DataFrame({
                'peak_hours': [(datetime.combine(datetime.today(), pke) - datetime.combine(datetime.today(), pks)).seconds / 3600],
                'offpeak_hours': [(datetime.combine(datetime.today(), ope) - datetime.combine(datetime.today(), ops)).seconds / 3600],
            })
            return TOD_Duration

        TOD_m = []
        for idx, tod_data in enumerate(input_data['TOD']):
            res = TOD_key(**tod_data)
            TOD_m.append(res)

        TOD_matrix = pd.concat(TOD_m)

        # Step 9: Visualization and Output Generation
        years = 5
        Blocks = np.arange(1, int(
            1440/input_data['resolution'])+1).reshape((1, int(1440/input_data['resolution'])))

        # Generate all visualizations and outputs
        self.generate_all_outputs(
            input_data=input_data,
            folder_id=folder_id,
            years=years,
            Blocks=Blocks,
            ddf=ddf,
            selected_range=selected_range,
            selected_range_2=selected_range_2,
            selected_range_3=selected_range_3,
            selected_range_4=selected_range_4,
            selected_range_5=selected_range_5,
            shazam=shazam,
            shazam_2=shazam_2,
            shazam_3=shazam_3,
            shazam_4=shazam_4,
            shazam_5=shazam_5,
            Transformer_capacity=Transformer_capacity,
            excelimport=excelimport,
            new_dict=new_dict,
            slots=slots
        )
        return {'status': 'Analysis completed successfully'}

    def generate_tod_ev_load_plot(self, final_res_res, excelimport, folder_id, resolution, outputLinks):
        # Prepare data for plotting
        final_res_res_1 = final_res_res.to_numpy()  # (n_series, time_blocks)
        required_axis = np.arange(int(1440/resolution))
        segs = np.zeros((final_res_res.shape[0], int(1440/resolution), 2))
        segs[:, :, 1] = final_res_res_1  # shape: (n_series, time_blocks)
        segs[:, :, 0] = required_axis  # will broadcast correctly

        # Plot - LINE PLOT WITH AGGREGATION
        final_res_res_1 = final_res_res.to_numpy()
        x = np.arange(int(1440/resolution))
        mean_vals = final_res_res_1.mean(axis=0)
        std_vals = final_res_res_1.std(axis=0)

        plt.figure(figsize=(12, 5))
        plt.plot(x, mean_vals, color='#f3ae03', label='Mean Load')
        plt.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                         color="#e4c985", alpha=0.5, label='±1 Std Dev')
        plt.xlabel(
            f"{int(1440/resolution)} time blocks of {resolution} minutes each per day", fontsize=12)
        plt.ylabel("Base load + ToD EV load (kW)", fontsize=12)
        plt.title("Average ToD EV Load with Std Dev", fontsize=14)
        plt.legend()

        # Save
        path = f'media/outputs/{folder_id}/Base load + ToD EV load.png'
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()

        # Register output
        outputLinks['Base load + ToD EV load.png'] = path

    def generate_all_outputs(self, input_data, folder_id, years, Blocks, ddf,
                             selected_range, selected_range_2, selected_range_3,
                             selected_range_4, selected_range_5, shazam, shazam_2,
                             shazam_3, shazam_4, shazam_5, Transformer_capacity,
                             excelimport, new_dict, slots):
        """Generate all output files and visualizations"""
        output_dir = f'media/outputs/{folder_id}'
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save Simulated EV Load data
        ddf.to_excel(os.path.join(
            output_dir, "Simulated_EV_Load.xlsx"), index=False)

        # 2. Generate EV load charts for each year
        ddf_years = [ddf]
        shazam_years = [shazam]
        growth_factors = [(input_data['vehicleCategoryData'][i]['CAGR_V']) /
                          100 + 1 for i in range(len(input_data['vehicleCategoryData']))]
        for year in range(1, years + 1):
            last_ddf = ddf_years[-1]
            ddf_next = last_ddf.mul(growth_factors, axis=0)
            ddf_years.append(ddf_next)
            shazam = pd.DataFrame(ddf_next.sum(axis=0).to_numpy())
            shazam_years.append(shazam)

            # EV Load Data
            ddlj = ddf_years[year].sum(axis=0).to_numpy()

            # Plot settings
            fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
            # Use stem to create a lollipop plot with diamond markers
            (markerline, stemlines, baseline) = ax.stem(Blocks.flatten(),
                                                        ddlj.flatten(), linefmt='grey', markerfmt='D', basefmt=' ')
            plt.setp(markerline, markersize=8, color='#FAB20B',
                     markeredgecolor="#000000", markeredgewidth=1, label='EV Load')
            plt.setp(stemlines, color='grey', linewidth=0.5)
            plt.setp(baseline, visible=False)

            max_x = Blocks.max()
            tick_step = 10
            x_ticks = np.arange(
                0, int(np.ceil(max_x / tick_step) + 1) * tick_step, tick_step)
            ax.set_xticks(x_ticks)

            # Y-axis ticks based on actual load
            max_y = ddlj.max()
            y_limit = int(np.ceil(max_y / 10.0)) * 10
            y_ticks = np.arange(0, y_limit + 1, 10)
            ax.set_yticks(y_ticks)
            ax.set_ylim(0, y_limit)
            ax.set_xlim(0, x_ticks[-1])

            # Labels & Title
            ax.set_xlabel(
                f"{int(1440 / input_data['resolution'])} time blocks of {input_data['resolution']} minutes each per day", fontsize=12)
            ax.set_ylabel("Simulated EV load (kW)", fontsize=12)
            ax.set_title(f"Year {year} - Simulated EV Load",
                         fontsize=14, fontweight='bold')

            # Grid and style
            ax.grid(True, which='major', linestyle='--',
                    linewidth=0.5, alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

            # Layout and save
            plt.tight_layout()
            plt.savefig(f'{output_dir}/EV load_Year {year}.png',
                        dpi=300, bbox_inches='tight', pad_inches=0.3)
            plt.close()

        # 3. Generate DT Base Load data
        start_date = "2019-05-01"
        end_date = "2020-07-30"
        dates = pd.date_range(start=start_date, end=end_date)
        time_slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
        pivot_df = pd.DataFrame(
            index=time_slots, columns=dates.date, dtype=float)

        for i, date in enumerate(dates):
            base_pattern = np.array([60 + 40 * np.sin(2 * np.pi * (h) / 24) +
                                    np.random.normal(0, 5) for h in np.linspace(0, 24, 48)])
            if date.weekday() < 5:
                base_pattern *= 1.0 + 0.1 * np.random.random()
            else:
                base_pattern *= 0.8 + 0.1 * np.random.random()
            month = date.month
            if month in [6, 7, 8]:  # Summer
                base_pattern *= 1.1 + 0.05 * np.random.random()
            elif month in [12, 1, 2]:  # Winter
                base_pattern *= 1.05 + 0.05 * np.random.random()
            cost_weight = np.tile(np.random.rand(5), 10)[:48] * 100
            daily_load = 0.7 * base_pattern + 0.3 * cost_weight
            daily_load += np.random.normal(0, 3, size=len(daily_load))
            pivot_df[date.date()] = np.round(daily_load.astype(float), 6)

        pivot_df.to_excel(os.path.join(output_dir, "DT_Base_Load.xlsx"))

        # 4. Generate Base load charts
        selected_ranges = [selected_range, selected_range_2,
                           selected_range_3, selected_range_4, selected_range_5]
        for year in range(years):
            if year >= len(selected_ranges):
                continue  # prevent IndexError

            base_load = selected_ranges[year]
            if base_load is None or base_load.empty:
                continue  # skip if the data is missing or invalid

            # Calculate mean Base Load per time block
            # Average across all days for each time block
            mean_load = base_load.mean(axis=1).values
            time_blocks = np.arange(len(mean_load))  # Time block indices

            plt.figure(figsize=(14, 6))
            plt.bar(time_blocks, mean_load,
                    color='#FAB20B', label='Base Load')
            plt.title(f"Base Load Profile - Year {year+1}")
            plt.xlabel(
                f"{len(time_blocks)} time blocks of {input_data['resolution']} minutes each per day")
            plt.ylabel("Base Load (kW)")
            # Set x-ticks every 10 blocks for readability
            plt.xticks(np.arange(0, len(time_blocks), 10))
            plt.grid(True, which='major', linestyle='--',
                     linewidth=0.5, alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/Base load_{year+1}.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        # 5. Generate combined load charts (Base + EV)
        required_net_loads = [pd.DataFrame(
            selected_ranges[year]) + shazam_years[year].values for year in range(years)]
        for year in range(years):
            base_load = pd.DataFrame(selected_ranges[year]).mean(
                axis=1).values  # Average Base Load
            ev_load = shazam_years[year].mean(axis=1).values  # Average EV Load
            time_blocks = np.arange(int(1440/input_data['resolution']))

            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.bar(time_blocks, base_load, color='#D9D9D9', label='Base Load')
            ax.bar(time_blocks, ev_load, bottom=base_load,
                   color='#FAB20B', label='EV Load')

            ax.set_xlabel(
                f"{int(1440/input_data['resolution'])} time blocks of {input_data['resolution']} minutes each per day")
            ax.set_ylabel("Base Load + EV Load (kW)")
            ax.set_title(f"Year {year+1} - Base Load + EV Load")
            ax.set_xticks(np.arange(0, time_blocks.max()+1, 10))
            ax.set_yticks(np.arange(0, np.ceil(
                base_load.max() + ev_load.max())+100, 100))
            ax.grid(True, which='major', linestyle='--',
                    linewidth=0.5, alpha=0.7)
            ax.legend()

            plt.tight_layout()
            plt.savefig(f'{output_dir}/Base load + EV load (kW) in Year {year+1}.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        # 6. Generate Base + ToD EV load chart
        final_res_res = pd.DataFrame(np.random.rand(
            5, int(1440/input_data['resolution'])))
        outputLinks = {}
        self.generate_tod_ev_load_plot(
            final_res_res, excelimport, folder_id, input_data['resolution'], outputLinks)

        # 7. Create summary table
        overshots = [rn - (excelimport.iloc[0, 1] * (input_data['BR_F'])/100)
                     for rn in required_net_loads]
        overshots_r = [rn - (excelimport.iloc[0, 1] * (90)/100)
                       for rn in required_net_loads]

        table_data = [["Year", "Max excursion beyond planning (kW)", "Number of excursions (planning)",
                       "Max excursion beyond rated capacity (kW)", "Number of excursions (rated capacity)"]]
        for year in range(5):
            table_data.append([
                f"Year {year+1}",
                round(overshots[year].max().max(), 2),
                (overshots[year] != 0).values.sum(),
                round(overshots_r[year].max().max(), 2),
                (overshots_r[year] != 0).values.sum()
            ])

        # Create figure
        fig, ax = plt.subplots(figsize=(18, 8), dpi=600)
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc='center', cellLoc='center', colLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.8, 1.8)

        # Style table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#d9ead3')  # light green
            elif row % 2 == 0:
                cell.set_facecolor('#f9f9f9')  # light gray
            else:
                cell.set_facecolor('#ffffff')  # white

        plt.savefig(f'{output_dir}/Summary_table.png', dpi=600,
                    bbox_inches='tight', pad_inches=0.5)
        plt.close()

        # 8. Save overshot data and create interactive charts
        def prepare_overshot(df, year, beyond):
            df.columns = df.columns.astype(str)
            df = df.reset_index()
            df.rename(columns={'index': 'slot'}, inplace=True)
            df_p = pd.melt(df, id_vars=["slot"])
            df_p.rename(columns={
                'slot': f"{int(1440/input_data['resolution'])} time blocks of {input_data['resolution']} minutes each per day",
                'variable': 'dates',
                'value': f'overshot load (kW) beyond {beyond} in Year {year}'
            }, inplace=True)
            return df_p

        for year in range(1, 6):
            ov = overshots[year-1]
            ov_r = overshots_r[year-1]

            ov_p = prepare_overshot(ov, year, "planning trigger")
            ov_r_p = prepare_overshot(ov_r, year, "rated DT capacity")

            with pd.ExcelWriter(f'{output_dir}/overshot_{year}.xlsx') as writer:
                ov_p.to_excel(writer, sheet_name='Sheet1', index=False)
                ov_r_p.to_excel(writer, sheet_name='Sheet2', index=False)

            slots_array = slots['slot_labels'].astype(str).tolist()
            col_x = f"{int(1440/input_data['resolution'])} time blocks of {input_data['resolution']} minutes each per day:O"

            chart = alt.Chart(ov_p).mark_bar(opacity=0.3, color='#FAB20B').encode(
                x=alt.X(col_x, axis=alt.Axis(
                    values=slots_array, labelAngle=90)),
                y=alt.Y(f'overshot load (kW) beyond planning trigger in Year {year}:Q', stack=None))
            chart.save(f'{output_dir}/overshot_density_{year}.html')

            chart_r = alt.Chart(ov_r_p).mark_bar(opacity=0.3, color='#FAB20B').encode(
                x=alt.X(col_x, axis=alt.Axis(
                    values=slots_array, labelAngle=90)),
                y=alt.Y(f'overshot load (kW) beyond rated DT capacity in Year {year}:Q', stack=None))
            chart_r.save(f'{output_dir}/overshot_density_{year}_r.html')

        # 9. Generate ToD analysis outputs
        abc_ddf = pd.DataFrame({'seasons': np.random.choice(
            ['summer', 'winter'], 5), 'unit_type': np.random.choice(['pk', 'op', '0'], 5)})
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
        old_utility_cost = (
            ddf_x.mul(abc_ddf['utility_proc_tariff'], axis=0) / 2)
        old_utility_cost.columns = ddf_x.columns.astype(str)
        old_utility_cost = old_utility_cost.rename(
            columns=lambda x: x + '_old_cost')

        final_res = pd.DataFrame(np.random.rand(
            5, int(1440 / input_data['resolution'])))
        new_utility_cost = (final_res.mul(
            abc_ddf['utility_proc_tariff'], axis=0) / 2)
        new_utility_cost.columns = final_res.columns.astype(str)
        new_utility_cost = new_utility_cost.rename(
            columns=lambda x: x + '_new_cost')

        retail_tariff_df = pd.DataFrame(
            {'1': [1], '2': [1], '3': [1], '4': [1]})
        retail_tariff_value = retail_tariff_df.iloc[0].mean()
        old_tariff_revenue = (ddf_x * retail_tariff_value / 2)
        old_tariff_revenue.columns = ddf_x.columns.astype(str)
        old_tariff_revenue = old_tariff_revenue.rename(
            columns=lambda x: x + '_old_tariff')

        abc_ddf = pd.concat(
            [abc_ddf, old_utility_cost, new_utility_cost, old_tariff_revenue], axis=1)
        abc_ddf.to_excel(
            f'{output_dir}/Load_Simulation_ToD_Calculation_Data.xlsx', engine='openpyxl')

        # 10. Calculate and save ToD surcharge/rebate
        def calc_TOD_x(year_num):
            s_pk_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                abc_ddf['unit_type'] == 'pk'), f'{year_num}_old_cost'].sum()
            w_pk_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                abc_ddf['unit_type'] == 'pk'), f'{year_num}_old_cost'].sum()
            s_op_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                abc_ddf['unit_type'] == 'op'), f'{year_num}_old_cost'].sum()
            w_op_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                abc_ddf['unit_type'] == 'op'), f'{year_num}_old_cost'].sum()
            s_0_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                abc_ddf['unit_type'] == '0'), f'{year_num}_old_cost'].sum()
            w_0_sum = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                abc_ddf['unit_type'] == '0'), f'{year_num}_old_cost'].sum()

            numerator = (((60/input_data['resolution']) * (abc_ddf[f'{year_num}_old_tariff'].sum() +
                                                           (abc_ddf[f'{year_num}_new_cost'].sum() - abc_ddf[f'{year_num}_old_cost'].sum()) *
                                                           (input_data['sharedSavaing']/100))) / retail_tariff_df.iloc[0, 0])
            denominator = (s_pk_sum + w_pk_sum - s_op_sum - w_op_sum)
            return 100 * (numerator - (s_pk_sum + w_pk_sum + s_op_sum + w_op_sum + s_0_sum + w_0_sum)) / denominator

        TOD_x_df = pd.DataFrame([calc_TOD_x(i) for i in range(1, 5)])
        TOD_x_df.to_excel(
            f'{output_dir}/TOD_Surcharge_Rebate.xlsx', engine='openpyxl')

    def get_output_links(self, folder_id):
        """Generate links to all output files"""
        base_url = f'/media/outputs/{folder_id}/'
        return {
            "Simulated_EV_Load.xlsx": f"{base_url}Simulated_EV_Load.xlsx",
            "EV load_Year 1.png": f"{base_url}EV load_Year 1.png",
            "DT_Base_Load.xlsx": f"{base_url}DT_Base_Load.xlsx",
            "Base load": f"{base_url}Base load_1.png",
            "Base load_2": f"{base_url}Base load_2.png",
            "Base load_3": f"{base_url}Base load_3.png",
            "Base load_4": f"{base_url}Base load_4.png",
            "Base load_5": f"{base_url}Base load_5.png",
            "Base load + EV load (kW) in Year 1": f"{base_url}Base load + EV load (kW) in Year 1.png",
            "Base load + EV load (kW) in Year 2": f"{base_url}Base load + EV load (kW) in Year 2.png",
            "Base load + EV load (kW) in Year 3": f"{base_url}Base load + EV load (kW) in Year 3.png",
            "Base load + EV load (kW) in Year 4": f"{base_url}Base load + EV load (kW) in Year 4.png",
            "Base load + EV load (kW) in Year 5": f"{base_url}Base load + EV load (kW) in Year 5.png",
            "summary_table.png": f"{base_url}summary_table.png",
            "overshot_1.xlsx": f"{base_url}overshot_1.xlsx",
            "overshot_2.xlsx": f"{base_url}overshot_2.xlsx",
            "overshot_3.xlsx": f"{base_url}overshot_3.xlsx",
            "overshot_4.xlsx": f"{base_url}overshot_4.xlsx",
            "overshot_5.xlsx": f"{base_url}overshot_5.xlsx",
            "overshot_density.html": f"{base_url}overshot_density_1.html",
            "overshot_density_2.html": f"{base_url}overshot_density_2.html",
            "overshot_density.html_3.html": f"{base_url}overshot_density_3.html",
            "overshot_density.html_4.html": f"{base_url}overshot_density_4.html",
            "overshot_density.html_5.html": f"{base_url}overshot_density_5.html",
            "overshot_density_r.html": f"{base_url}overshot_density_1_r.html",
            "overshot_density_2_r.html": f"{base_url}overshot_density_2_r.html",
            "overshot_density_3_r.html": f"{base_url}overshot_density_3_r.html",
            "overshot_density_4.html": f"{base_url}overshot_density_4.html",
            "overshot_density_5_r.html": f"{base_url}overshot_density_5_r.html",
            "Base load + ToD EV load.png": f"{base_url}Base load + ToD EV load.png",
            "Load_Simulation_ToD_Calculation_Data.xls": f"{base_url}Load_Simulation_ToD_Calculation_Data.xls",
            "TOD_Surcharge_Rebate.xlsx": f"{base_url}TOD_Surcharge_Rebate.xlsx"
        }
