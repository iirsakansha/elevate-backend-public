from rest_framework import permissions, status
import math
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

            # Handle uploaded file (isLoadSplitFile)
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
                os.makedirs(os.path.join(
                    fs.location, upload_folder), exist_ok=True)
                filename = fs.save(f"media/FileUpload/{file.name}", file)
                file_path = fs.path(filename)
                all_data['isLoadSplitFile'] = file_path

            # Process categories
            try:
                processed_data, created_load_categories, created_vehicle_categories = self.process_categories(
                    all_data)
                all_data.update(processed_data)
            except Exception as e:
                return Response({'error': f"process_categories failed: {str(e)}"}, status=500)

            # Serialize and save instance
            try:
                serializer = EvAnalysisSerializer(data=all_data)
                if not serializer.is_valid():
                    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                ev_instance = serializer.save()
            except Exception as e:
                return Response({'error': f"Serializer failed: {str(e)}"}, status=500)

            # Prepare analysis data
            analysis_data = self.prepare_analysis_data(ev_instance)

            # Run analysis and get raw data
            results = self.run_full_analysis(
                analysis_data, str(ev_instance.id))

            # Clean up temporary data
            self.cleanup_temporary_data(
                created_load_categories, created_vehicle_categories)

            return Response(results, status=status.HTTP_201_CREATED)

        except Exception as e:
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

        load_categories = all_data.get('categoryData', [])
        for idx, category_data in enumerate(load_categories[:6], start=1):
            category = LoadCategoryModel.objects.create(**category_data)
            created_load_categories.append(category)
            processed_data[f'loadCategory{idx}'] = category.id

        vehicle_categories = all_data.get('vehicleCategoryData', [])
        for idx, vehicle_data in enumerate(vehicle_categories[:5], start=1):
            vehicle = vehicleCategoryModel.objects.create(**vehicle_data)
            created_vehicle_categories.append(vehicle)
            processed_data[f'vehicleCategoryData{idx}'] = vehicle.id

        processed_data['loadCategory'] = len(load_categories)
        processed_data['numOfvehicleCategory'] = len(vehicle_categories)

        return processed_data, created_load_categories, created_vehicle_categories

    def cleanup_temporary_data(self, created_load_categories, created_vehicle_categories):
        """Delete all temporary category data created during processing"""
        try:
            for category in created_load_categories:
                try:
                    category.delete()
                except Exception as e:
                    print(f"Error deleting load category {category.id}: {e}")
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
                'rowlimit_xl': 1000000,
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
        """Run the complete analysis pipeline and return raw data"""
        # Step 1: Load forecast for each vehicle category
        def load_forecast(n, f, c, p, e, r, k, l, g, h, s, u, rowlimit_xl, CAGR_V, tariff):
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
            dummy_transpose = dummy.transpose()
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
            return Blo_load_sec.tolist()

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

        resolution = input_data['resolution']
        time_blocks_per_day = int(1440 / resolution)
        total_data_points = len(abc)
        complete_days = total_data_points // time_blocks_per_day

        if total_data_points % time_blocks_per_day != 0:
            abc_trimmed = abc.iloc[:complete_days * time_blocks_per_day]
            print(
                f"Warning: Data trimmed from {total_data_points} to {len(abc_trimmed)} points to fit {complete_days} complete days")
        else:
            abc_trimmed = abc

        try:
            load_extract = pd.DataFrame(np.reshape(
                abc_trimmed.to_numpy(), (complete_days, time_blocks_per_day)))
            load_extract = load_extract.T
        except ValueError as e:
            print(f"Reshape error: {e}")
            raise

        if len(labels) != load_extract.shape[1]:
            if len(labels) > load_extract.shape[1]:
                labels = labels[:load_extract.shape[1]]
                print(f"Trimmed labels to {len(labels)} to match data columns")
            else:
                additional_needed = load_extract.shape[1] - len(labels)
                labels_to_repeat = labels[-additional_needed:] if additional_needed <= len(
                    labels) else labels
                labels = np.concatenate(
                    [labels, labels_to_repeat[:additional_needed]])
                print(
                    f"Extended labels to {len(labels)} to match data columns")

        final_load = load_extract.copy()
        final_load.columns = labels

        # Project Loads Over Years
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
        growth_factors = [(input_data['vehicleCategoryData'][i]['CAGR_V']) /
                          100 + 1 for i in range(len(input_data['vehicleCategoryData']))]
        ddf2 = ddf.mul(growth_factors, axis=0)
        shazam_2 = pd.DataFrame((ddf2.sum(axis=0)).to_numpy())
        ddf3 = ddf2.mul(growth_factors, axis=0)
        shazam_3 = pd.DataFrame((ddf3.sum(axis=0)).to_numpy())
        ddf4 = ddf3.mul(growth_factors, axis=0)
        shazam_4 = pd.DataFrame((ddf4.sum(axis=0)).to_numpy())
        ddf5 = ddf4.mul(growth_factors, axis=0)
        shazam_5 = pd.DataFrame((ddf5.sum(axis=0)).to_numpy())

        # Time-of-Day (ToD) Analysis
        def TOD_key(pks, pke, sx, ops, ope, rb):
            pke = pke.replace("24:00", "23:59")
            ope = ope.replace("24:00", "23:59")
            FMT = '%H:%M'
            pks = datetime.strptime(str(pks), FMT).time()
            pke = datetime.strptime(str(pke), FMT).time()
            ops = datetime.strptime(str(ops), FMT).time()
            ope = datetime.strptime(str(ope), FMT).time()
            TOD_Duration = {
                'peak_hours': (datetime.combine(datetime.today(), pke) - datetime.combine(datetime.today(), pks)).seconds / 3600,
                'offpeak_hours': (datetime.combine(datetime.today(), ope) - datetime.combine(datetime.today(), ops)).seconds / 3600,
            }
            return TOD_Duration

        TOD_m = []
        for idx, tod_data in enumerate(input_data['TOD']):
            res = TOD_key(**tod_data)
            TOD_m.append(res)
        TOD_matrix = pd.DataFrame(TOD_m)

        # Generate raw data for outputs
        output_data = self.generate_all_outputs(
            input_data=input_data,
            folder_id=folder_id,
            years=5,
            Blocks=np.arange(
                1, int(1440/input_data['resolution'])+1, 1).tolist(),
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

        return output_data

    def generate_tod_ev_load_plot(self, final_res_res, excelimport, resolution):
        """Generate raw data for ToD EV load plot"""
        final_res_res_1 = final_res_res.to_numpy()
        mean_vals = final_res_res_1.mean(axis=0).tolist()
        std_vals = final_res_res_1.std(axis=0).tolist()
        time_blocks = np.arange(int(1440/resolution)).tolist()
        return {
            'time_blocks': time_blocks,
            'mean_load': mean_vals,
            'std_dev': std_vals
        }

    def generate_all_outputs(self, input_data, folder_id, years, Blocks, ddf,
                             selected_range, selected_range_2, selected_range_3,
                             selected_range_4, selected_range_5, shazam, shazam_2,
                             shazam_3, shazam_4, shazam_5, Transformer_capacity,
                             excelimport, new_dict, slots):
        """Generate raw data for all outputs"""
        output_data = {}

        # 1. Simulated EV Load data
        output_data['Simulated_EV_Load'] = ddf.values.tolist()

        # 2. EV load data for each year
        ddf_years = [ddf]
        shazam_years = [shazam]
        growth_factors = [(input_data['vehicleCategoryData'][i]['CAGR_V']) /
                          100 + 1 for i in range(len(input_data['vehicleCategoryData']))]
        ev_loads = []
        for year in range(1, years + 1):
            last_ddf = ddf_years[-1]
            ddf_next = last_ddf.mul(growth_factors, axis=0)
            ddf_years.append(ddf_next)
            shazam = pd.DataFrame(ddf_next.sum(axis=0).to_numpy())
            shazam_years.append(shazam)
            ev_loads.append({
                f'Year {year}': ddf_next.sum(axis=0).to_numpy().flatten().tolist()
            })

        output_data['EV_Load'] = ev_loads

        # 3. Base Load data
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
            if month in [6, 7, 8]:
                base_pattern *= 1.1 + 0.05 * np.random.random()
            elif month in [12, 1, 2]:
                base_pattern *= 1.05 + 0.05 * np.random.random()
            cost_weight = np.tile(np.random.rand(5), 10)[:48] * 100
            daily_load = 0.7 * base_pattern + 0.3 * cost_weight
            daily_load += np.random.normal(0, 3, size=len(daily_load))
            pivot_df[date.date()] = np.round(daily_load.astype(float), 6)
        output_data['DT_Base_Load'] = pivot_df.values.tolist()

        # 4. Base load charts data
        selected_ranges = [selected_range, selected_range_2,
                           selected_range_3, selected_range_4, selected_range_5]
        base_loads = []
        for year in range(years):
            if year >= len(selected_ranges):
                continue
            base_load = selected_ranges[year]
            if base_load is None or base_load.empty:
                continue
            mean_load = base_load.mean(axis=1).values.tolist()
            base_loads.append({
                f'Year {year+1}': mean_load
            })
        output_data['Base_Load'] = base_loads

        # 5. Combined load (Base + EV)
        required_net_loads = [pd.DataFrame(
            selected_ranges[year]) + shazam_years[year].values for year in range(years)]
        combined_loads = []
        for year in range(years):
            base_load = pd.DataFrame(selected_ranges[year]).mean(
                axis=1).values.tolist()
            ev_load = shazam_years[year].mean(axis=1).values.tolist()
            combined_loads.append({
                f'Year {year+1}': {
                    'base_load': base_load,
                    'ev_load': ev_load
                }
            })
        output_data['Base_EV_Load'] = combined_loads

        # 6. Base + ToD EV load
        final_res_res = pd.DataFrame(np.random.rand(
            5, int(1440/input_data['resolution'])))
        output_data['Base_ToD_EV_Load'] = self.generate_tod_ev_load_plot(
            final_res_res, excelimport, input_data['resolution'])

        # 7. Summary table
        overshots = [rn - (excelimport.iloc[0, 1] * (input_data['BR_F'])/100)
                     for rn in required_net_loads]
        overshots_r = [rn - (excelimport.iloc[0, 1] * (90)/100)
                       for rn in required_net_loads]
        table_data = []
        for year in range(5):
            table_data.append({
                'Year': f'Year {year+1}',
                'Max_excursion_planning': round(overshots[year].max().max(), 2),
                'Num_excursions_planning': (overshots[year] != 0).values.sum(),
                'Max_excursion_rated': round(overshots_r[year].max().max(), 2),
                'Num_excursions_rated': (overshots_r[year] != 0).values.sum()
            })
        output_data['Summary_Table'] = table_data

        # 8. Overshot data
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
            overshot_data.append({
                f'Year {year}': {
                    'planning': ov_p.to_dict('records'),
                    'rated': ov_r_p.to_dict('records')
                }
            })
        output_data['Overshot'] = overshot_data

        # 9. ToD analysis
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
        output_data['Load_Simulation_ToD_Calculation'] = abc_ddf.to_dict(
            'records')

        # 10. ToD surcharge/rebate
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

        output_data['TOD_Surcharge_Rebate'] = [
            calc_TOD_x(i) for i in range(1, 5)]

        return output_data
