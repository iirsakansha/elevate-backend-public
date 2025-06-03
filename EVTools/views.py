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
# EV Analysis


class EvAnalysisView(APIView):
    # parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.IsAuthenticated, ]

    def post(self, request, *args, **kwargs):
        allData = request.data
        cat = 1
        LoadCategoryList = {'loadCategory1': None, 'loadCategory2': None,
                            'loadCategory3': None, 'loadCategory4': None, 'loadCategory5': None, 'loadCategory6': None}
        print(len(allData['categoryData']), "len of CatDta")
        for loadCatData in allData.pop('categoryData'):
            print(loadCatData, "loadCatNum")
            LoadCategory = LoadCategoryModelSerializer(data=loadCatData)
            if LoadCategory.is_valid():
                LoadCategory.save()
                LoadCategoryList[f'loadCategory{cat}'] = LoadCategory.data['id']
                cat = cat + 1

        v_cat = 1
        vehicleCategoryList = {'vehicleCategoryData1': None, 'vehicleCategoryData2': None,
                               'vehicleCategoryData3': None, 'vehicleCategoryData4': None, 'vehicleCategoryData5': None}
        for vehicleData in allData.pop('vehicleCategoryData'):
            vehicleCategory = vehicleCategoryModelSerializer(data=vehicleData)
            if vehicleCategory.is_valid():
                vehicleCategory.save()
                vehicleCategoryList[f'vehicleCategoryData{v_cat}'] = vehicleCategory.data['id']
                v_cat = v_cat + 1
            else:
                print(vehicleCategory.errors, "Errors - vehical Cat")

        allData.update(LoadCategoryList)
        allData.update(vehicleCategoryList)

        EvAnalysis = EvAnalysisSerializer(data=allData)
        if EvAnalysis.is_valid():
            EvAnalysis.save()
            # print(EvAnalysis.data, "Srializer Data")

            begin = time.time()
            int_time = time.time()
            # ___________________________________________________________________________________________

            folderId = EvAnalysis.data['id']
            userName = EvAnalysis.data['user_name']
            os.makedirs(f'media/outputs/{folderId}')
            outputLinks = {}
            outputLinks['id'] = folderId

            EvDataDBObjects = evAnalysis.objects.get(id=int(folderId))
            # print(EvDataDBObjects, "Table Data")

            # <--strat row data-->
            resolution = EvDataDBObjects.resolution
            # print(resolution, "Resolution")

            BR_F = EvDataDBObjects.BR_F
            # print(BR_F, "BR_F")

            isLoadSplitFile = EvDataDBObjects.isLoadSplitFile
            isLoadSplitFile = isLoadSplitFile[1:]
            # print(isLoadSplitFile, "isLoadSplitFileUrl")

            shared_savings = EvDataDBObjects.sharedSavaing
            # print(shared_savings, "sharedSavaing")
            seasons = pd.DataFrame({'start_date': [EvDataDBObjects.date1_start, EvDataDBObjects.date1_end],
                                    'end_date': [EvDataDBObjects.date2_start, EvDataDBObjects.date2_end]})
            # print(seasons, "seasons")
            numOfvehicleCategory = EvDataDBObjects.numOfvehicleCategory
            # print(numOfvehicleCategory, "Num Of Vahicle Cat")

            # print("sum_pk_cost")
            sum_pk_cost = EvDataDBObjects.sum_pk_cost
            # print(sum_pk_cost, "sum_pk_cost")
            sum_0_cost = EvDataDBObjects.sum_zero_cost
            # print(sum_0_cost, "sum_0_cost")
            sum_op_cost = EvDataDBObjects.sum_op_cost
            # print(sum_op_cost, "sum_op_cost")
            win_pk_cost = EvDataDBObjects.win_pk_cost
            # print(win_pk_cost, "win_pk_cost")
            win_0_cost = EvDataDBObjects.win_zero_cost
            # print(win_0_cost, "win_0_cost")
            win_op_cost = EvDataDBObjects.win_op_cost
            # print(win_op_cost, "win_op_cost")

            TOD = {0: {'pks': EvDataDBObjects.s_pks, 'pke': EvDataDBObjects.s_pke, 'sx': EvDataDBObjects.s_sx, 'ops': EvDataDBObjects.s_ops, 'ope': EvDataDBObjects.s_ope, 'rb': EvDataDBObjects.s_rb},
                   1: {'pks': EvDataDBObjects.w_pks, 'pke': EvDataDBObjects.w_pke, 'sx': EvDataDBObjects.w_sx, 'ops': EvDataDBObjects.w_ops, 'ope': EvDataDBObjects.w_ope, 'rb': EvDataDBObjects.w_rb}}

            if EvDataDBObjects.isLoadSplit == "no":
                try:
                    excelimport = pd.read_excel(isLoadSplitFile, header=None)

                    numOfvehicleCategorys = EvDataDBObjects.numOfvehicleCategory
                    my_blank_dict = {}
                    for numOfvehicleCategory in range(1, numOfvehicleCategorys + 1):
                        vehicleCategoryData = vehicleCategoryModel.objects.get(
                            id=getattr(EvDataDBObjects, f"vehicleCategoryData{numOfvehicleCategory}_id"))
                        my_blank_dict[numOfvehicleCategory] = {'n': vehicleCategoryData.n, 'f': vehicleCategoryData.f, 'c': vehicleCategoryData.c, 'p': vehicleCategoryData.p, 'e': vehicleCategoryData.e, 'r': vehicleCategoryData.r,
                                                               'k': vehicleCategoryData.k, 'l': vehicleCategoryData.l, 'g': vehicleCategoryData.g, 'h': vehicleCategoryData.h, 's': vehicleCategoryData.s, 'u': vehicleCategoryData.u, 'rowlimit_xl': vehicleCategoryData.rowlimit_xl, 'CAGR_V': vehicleCategoryData.CAGR_V, 'tariff': vehicleCategoryData.baseElectricityTariff}
                    my_dict = my_blank_dict

                    # my_dict = {1: {'n': 500, 'f': 1, 'c': 38, 'p': 3.3, 'e': 95, 'r': 300, 'k': 1200, 'l': 120, 'g': 10, 'h': 3, 's': 95, 'u': 2, 'rowlimit_xl': 2000000, 'CAGR_V': 5, 'tariff': 5},
                    #            2: {'n': 500, 'f': 1, 'c': 5, 'p': 0.33, 'e': 95, 'r': 40, 'k': 900, 'l': 120, 'g': 25, 'h': 3, 's': 95, 'u': 2, 'rowlimit_xl': 2000000, 'CAGR_V': 5, 'tariff': 6},
                    #            3: {'n': 250, 'f': 1, 'c': 8, 'p': 1.32, 'e': 95, 'r': 120, 'k': 420, 'l': 120, 'g': 25, 'h': 3, 's': 95, 'u': 2, 'rowlimit_xl': 2000000, 'CAGR_V': 5, 'tariff': 8},
                    #            4: {'n': 250, 'f': 1, 'c': 8, 'p': 1.32, 'e': 95, 'r': 120, 'k': 1080, 'l': 120, 'g': 25, 'h': 3, 's': 95, 'u': 2, 'rowlimit_xl': 2000000, 'CAGR_V': 5, 'tariff': 8}}

                    loadCategorys = EvDataDBObjects.loadCategory
                    new_blank_dict = {1: {'com': 0, 'ind': 0, 'res': 0, 'pub': 0, 'agr': 0, 'other': 0},
                                      2: {'com': 0, 'ind': 0, 'res': 0, 'pub': 0, 'agr': 0, 'other': 0}, }
                    # 1 is load share category wise, 2 is CAGR of category wise sales, and 3 is base electricity tariff
                    loadcategoryData = {}

                    for NumofCategory in range(1, loadCategorys+1):
                        loadcategoryData[NumofCategory] = LoadCategoryModel.objects.get(
                            id=getattr(EvDataDBObjects, f"loadCategory{NumofCategory}_id"))
                    for i in range(0, 2):
                        for data in loadcategoryData.values():
                            cat = data.category[0:3] if data.category != "others" else data.category[0:5]
                            if i == 0:
                                new_blank_dict[i+1][cat] = data.specifySplit
                            elif i == 1:
                                new_blank_dict[i+1][cat] = data.salesCAGR
                    new_dict = new_blank_dict

                    # new_dict = {1: {'com': 48, 'ind': 0, 'res': 35, 'pub': 0, 'agr': 0, 'other': 17},
                    #             2: {'com': 3, 'ind': 3, 'res': 3, 'pub': 3, 'agr': 3, 'other': 3}}

                    # start algorithm
                    my_dict_df = pd.DataFrame(my_dict).T
                    retail_tariff_df = pd.DataFrame(
                        my_dict_df['tariff'].values.reshape(1, -1))
                    retail_tariff_df.columns += 1
                    retail_tariff_df.columns = retail_tariff_df.columns.astype(
                        str)

                    def load_forecast(n, f, c, p, e, r, k, l, g, h, s, u, rowlimit_xl, CAGR_V, tariff):
                        begin = time.time()
                        # Calcuting total charges of the given vehicle category in a single day
                        Total_Charges = n*f
                        Str1 = 'Total charges per day'

                        # Creating the timeline of the day in blocks
                        Blocks = numpy.arange(1, int(1440/resolution)+1,
                                              1).reshape((1, int(1440/resolution)))
                        Str2 = 'Timeline in chosen blocks of minutes in a day'
                        bf = pd.DataFrame(Blocks)

                        Str3 = 'Normally distributing total charges across timeline with chosen mean and standard deviation'
                        ex1 = numpy.arange(Blocks.min(), Blocks.max()+1, 1)
                        ex1f = pd.DataFrame(ex1)
                        mu = math.ceil(k/resolution)
                        sigma = math.ceil(l/resolution)
                        Block_Charges = Total_Charges * \
                            (ss.norm.pdf(ex1, mu, sigma))  # the normal pdf
                        bcf = pd.DataFrame(Block_Charges)
                        Str4 = 'Charges at each block'

                        # Reshaping 'Block charges' into a single column array
                        Block_Charges_Column = numpy.reshape(
                            Block_Charges, (Blocks.max(), 1))
                        Str5 = Str4+', reshaped into a single column'
                        bccf = pd.DataFrame(Block_Charges_Column)

                        Total_Simulated_Block_Charges = sum(Block_Charges)

                        # Listing out all the possible 'distances travelled the previous day'
                        Kilometers = numpy.arange(0, r+1, 1).reshape((1, r+1))
                        Str6 = 'Possible distances travelled the previous day'
                        kf = pd.DataFrame(Kilometers)

                        # Listing out the proportional StartingSOC
                        StartingSOC = 100*(1-(Kilometers/r))
                        Str7 = 'StartingSOC'
                        ssocf = pd.DataFrame(StartingSOC)

                        Str8 = 'Normally distributing the "distance travelled the previous day" possibilities'
                        ex2 = numpy.arange(0, r+1, 1).reshape(1, r+1)
                        ex2f = pd.DataFrame(ex2)
                        mu2 = g
                        sigma2 = h
                        Prev_Distance_Prob = ss.norm.pdf(
                            ex2, mu2, sigma2)  # the normal pdf
                        Str9 = 'Probabilities of distances travelled the previous day'
                        pdstf = pd.DataFrame(Prev_Distance_Prob)

                        # Calculating vehicles with various charging start times and distances travelled the previous day
                        ATD = numpy.dot(Block_Charges_Column,
                                        Prev_Distance_Prob)
                        Str10 = 'Matrix of vehicles with "rows = various charging start times" and "columns = distances travelled the previous day"'
                        atdf = pd.DataFrame(ATD)
                        Total_Simulated_Block_Charges_2 = numpy.sum(ATD)

                        # Listing out the EndingSOCs
                        EndingSOC = numpy.arange(0, 101, 1).reshape(1, 101)
                        Str11 = 'Possible State of Charges at Charging Ending Time'
                        esocf = pd.DataFrame(EndingSOC)

                        # normally distributing the "EndingSOC" possibilities
                        mu3 = s
                        sigma3 = u
                        EndingSOC_Prob = ss.norm.pdf(
                            EndingSOC, mu3, sigma3)  # the normal pdf
                        Str13 = 'Probabilities of EndingSOC'
                        # Listing out all the lognormal values of EndingSOC for reference
                        esocpf = pd.DataFrame(EndingSOC_Prob)
                        goal = numpy.sum(EndingSOC_Prob)

                        # To facilitate matrix multiplications, creating a StartingSOC matrix by replicating StartingSOC_Prob 1440*(r+1) times as rows
                        dummy = numpy.tile(StartingSOC, (101, 1))
                        Str14 = 'Dummy'
                        dumf = pd.DataFrame(dummy)

                        dummy_transpose = (dummy.transpose())
                        Str15 = 'Dummy Transpose'
                        dumtf = pd.DataFrame(dummy_transpose)

                        StartingSOC_Matrix = numpy.tile(
                            dummy_transpose, (int(1440/resolution), 1))
                        Str16 = 'StartingSOC Matrix'
                        ssocmf = pd.DataFrame(StartingSOC_Matrix)

                        # reshape the StartSOC into a column, create a matrix with 101 such columns
                        # append existing range 1440-1 times
                        Str17 = 'EndingSOC Matrix'
                        EndingSOC_Matrix = numpy.tile(
                            EndingSOC, ((int(1440/resolution))*(r+1), 1))
                        esocmf = pd.DataFrame(EndingSOC_Matrix)

                        EndingSOC_Prob_Matrix = numpy.tile(
                            EndingSOC_Prob, ((int(1440/resolution))*(r+1), 1))
                        Str18 = 'EndingSOC Probabilities Matrix'
                        esocpmf = pd.DataFrame(EndingSOC_Prob_Matrix)

                        # Reshaping 'ATD' into a single column array
                        ATD_Column = numpy.reshape(
                            ATD, ((int(1440/resolution))*(r+1), 1))
                        Str19 = 'Reshaping into a single column array the "'+Str10+'"'
                        atdcf = pd.DataFrame(ATD_Column)

                        Veh_All_Comb = ATD_Column * EndingSOC_Prob_Matrix
                        Str20 = 'Vehicles under various charging start times, distances travelled the previous day, StartingSOCs and EndingSOCs'
                        vacf = pd.DataFrame(Veh_All_Comb)

                        # Calculating theoretical Charging Duration across all possibilities
                        # Multiplying with 2 as opposed to 60 minutes, because we are analysing for 30 min blocks
                        Charging_Duration = ((60*c/resolution)/(p*e)) * \
                            (EndingSOC_Matrix-StartingSOC_Matrix)
                        # Removing negative values here for now. But if Vehicle to grid discharge is considered then this step will need to change to retain negative values for discharge and positive for charge
                        Charging_Duration_P = numpy.where(
                            Charging_Duration < 0, 0, Charging_Duration)
                        Str21 = 'Theoretical Charging Duration across all possibilities'
                        cdpf = pd.DataFrame(Charging_Duration_P)

                        superdummy = Veh_All_Comb.reshape(
                            Veh_All_Comb.shape[0]*Veh_All_Comb.shape[1], 1)
                        Str22 = 'Reshaping into a single column, the '+Str20
                        sdf = pd.DataFrame(superdummy)

                        megadummy = Charging_Duration_P.reshape(
                            Charging_Duration_P.shape[0]*Charging_Duration_P.shape[1], 1)
                        Str23 = 'Reshaping into a single column, the '+Str21
                        # cannot be printed in a single excel sheet. Will be split up in the following cells to accomodate in multiple excel sheets
                        mdf = pd.DataFrame(megadummy)

                        Str24 = 'Excel permits only a maximum of 1,048,576 rows per sheet. We are using only ' + \
                            str(rowlimit_xl)
                        req_xl_file_number = math.ceil(
                            Charging_Duration_P.shape[0]*Charging_Duration_P.shape[1]/rowlimit_xl)
                        Str25 = 'Required excel files (with 1 sheet each) to display results'

                        Str26 = 'Splitting ('+Str20 + \
                            ') into manageable excel sheets'
                        # each new arrays will automatically be named in serise with the parent name
                        Veh_Output = numpy.array_split(
                            superdummy, req_xl_file_number)
                        Str31 = 'Splitting ('+Str21 + \
                            ') into manageable excel sheets'
                        # each new arrays will automatically be named in serise with the parent name
                        Output = numpy.array_split(
                            megadummy, req_xl_file_number)

                        Blocks_column = Blocks.reshape(int(1440/resolution), 1)
                        Blocks_column_iter = numpy.repeat(
                            Blocks_column, (r+1)*101).reshape((int(1440/resolution))*(r+1)*101, 1)
                        Str36 = 'Blocks_column_iter'
                        Str37 = 'Splitting ('+Str36 + \
                            ') into manageable excel sheets'
                        Blo_ref = numpy.array_split(
                            Blocks_column_iter, req_xl_file_number)

                        Int_output = Output[0].astype(int)
                        Float_diff = numpy.subtract(Output[0], Int_output)
                        Roll_d = [numpy.ones(i, dtype=numpy.uint8)
                                  for i in Int_output]
                        vf = pd.DataFrame(Veh_Output[0])
                        rld = pd.DataFrame(Roll_d)
                        brf = pd.DataFrame(Blo_ref[0])
                        Roll_i = [[Roll_d[i] * j for j in sub]
                                  for i, sub in enumerate(Veh_Output[0].tolist())]
                        Rol = pd.DataFrame(Roll_i)
                        Dlo_roll = [list(chain(*x)) for x in Roll_i]
                        Dlof = pd.DataFrame(Dlo_roll)
                        Zero_list = [numpy.zeros(i-1, dtype=numpy.uint8)
                                     for i in Blo_ref[0]]
                        zf = pd.DataFrame(Zero_list)
                        Float_mul_list = Veh_Output[0]*Float_diff
                        Fmf = pd.DataFrame(Float_mul_list)
                        Float_list = Float_mul_list.tolist()
                        Joint = list(list(x)
                                     for x in zip(Zero_list, Dlo_roll, Float_list))
                        Jf = pd.DataFrame(Joint)
                        Blo_roll = [list(chain(*x)) for x in Joint]
                        Blof = pd.DataFrame(Blo_roll)
                        Blo_sum_linear = list(
                            map(sum, itertools.zip_longest(*Blo_roll, fillvalue=0)))
                        Blosf = pd.DataFrame(Blo_sum_linear)
                        Split_points = [int(1440/resolution)]
                        Split_list = [Blo_sum_linear[i: j]
                                      for i, j in zip([0]+Split_points, Split_points + [None])]
                        Splf_0 = pd.DataFrame(Split_list[0])
                        Splf_1 = pd.DataFrame(Split_list[1])
                        Blo_sec = numpy.array(list(map(sum, itertools.zip_longest(
                            Split_list[0], Split_list[1], fillvalue=0))))
                        Blo_sec_f = pd.DataFrame(Blo_sec)
                        Blo_load_sec = ((p)*(Blo_sec)).reshape(1,
                                                               int(1440/resolution))
                        Blo_load_sec_f = pd.DataFrame(Blo_load_sec)
                        print(time.time() - begin, "calling Function")
                        return Blo_load_sec_f.values

                    dracula = list()
                    for i, j in my_dict.items():
                        res = load_forecast(**j)
                        dracula.append(res)
                    ddf = pd.DataFrame(numpy.concatenate(dracula))

                    ddf.to_excel(
                        f'media/outputs/{folderId}/Simulated_EV_Load.xlsx')
                    outputLinks['Simulated_EV_Load.xlsx'] = f'media/outputs/{folderId}/Simulated_EV_Load.xlsx'

                    ddlj = (ddf.sum(axis=0)).to_numpy()

                    # pd.DataFrame(ddlj).to_excel('ddlj.xlsx')

                    Blocks = numpy.arange(1, int(1440/resolution)+1,
                                          1).reshape((1, int(1440/resolution)))
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(Blocks.flatten(), ddlj.flatten())
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day")
                    plt.ylabel("Simulated EV load in kW")
                    plt.grid(True)
                    plt.xticks(rotation=90)
                    plt.savefig(f'media/outputs/{folderId}/EV load_Year 1.png',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks['EV load_Year 1.png'] = f'media/outputs/{folderId}/EV load_Year 1.png'
                    # plt.show()

                    DT = pd.DataFrame(numpy.tile(
                        excelimport.iloc[0, 1], 406*(int(1440/resolution))))
                    DT.rename(columns={"0": "DT"})
                    # DT.to_excel('DT.xlsx')

                    Source = excelimport.iloc[4:, :]
                    Source.columns = ['Meter.No', 'datetime_utc', 'Active_B_PH', 'Active_Y_PH',
                                      'Active_R_PH', 'Reactive_B_PH', 'Reactive_Y_PH', 'Reactive_R_PH', 'VBV', 'VYV', 'VRV']
                    Source = Source.reset_index(drop=True)

                    # Calculate Active Power without mentioning title for ease of future representation
                    Source[''] = (((Source.Active_B_PH * Source.VBV)+(Source.Active_Y_PH *
                                                                      Source.VYV)+(Source.Active_R_PH * Source.VRV))/1000)
                    Source['datetime_utc'] = pd.to_datetime(
                        Source['datetime_utc'])

                    Source['date'] = Source['datetime_utc'].dt.date
                    labels = Source['datetime_utc'].dt.date.unique()
                    slots = pd.DataFrame(
                        Source['datetime_utc'].dt.time.unique())
                    slots.columns = ['slot_labels']

                    Transformer_capacity = pd.DataFrame(numpy.repeat(
                        (excelimport.iloc[0, 1]*(BR_F)/100), 1440/resolution))
                    Transformer_capacity.columns = [
                        'Transformer safety planning trigger']
                    Transformer_capacity['100% Transformer rated capacity'] = numpy.repeat(
                        excelimport.iloc[0, 1]*(100/100), 1440/resolution)

                    Source.set_index('datetime_utc', inplace=True)
                    Calculated_load = Source.drop(['Meter.No', 'Active_B_PH', 'Active_Y_PH', 'Active_R_PH',
                                                   'Reactive_B_PH', 'Reactive_Y_PH', 'Reactive_R_PH', 'VBV', 'VYV', 'VRV'], axis=1)
                    Calculated_load.index.name = None
                    abc = pd.DataFrame(
                        Calculated_load[''].astype(float).values)
                    load_extract = pd.DataFrame(numpy.reshape(
                        abc.to_numpy(), (-1, int(1440/resolution))))
                    final_load = load_extract.T
                    final_load.columns = labels
                    final_load.to_excel(
                        f'media/outputs/{folderId}/DT_Base_Load.xlsx')
                    outputLinks['DT_Base_Load.xlsx'] = f'media/outputs/{folderId}/DT_Base_Load.xlsx'
                    selected_range = final_load.iloc[:, : 406].copy()

                    selected_range_2 = (selected_range/100)*(((new_dict[1]['com'])*(1+(new_dict[2]['com'])/100))+((new_dict[1]['ind'])*(1+(new_dict[2]['ind'])/100))+((new_dict[1]['res'])*(
                        1+(new_dict[2]['res'])/100))+((new_dict[1]['pub'])*(1+(new_dict[2]['pub'])/100))+((new_dict[1]['agr'])*(1+(new_dict[2]['agr'])/100))+((new_dict[1]['other'])*(1+(new_dict[2]['other'])/100)))
                    selected_range_2.columns = selected_range_2.columns + \
                        relativedelta(years=1)
                    selected_range_3 = (selected_range_2/100)*(((new_dict[1]['com'])*(1+(new_dict[2]['com'])/100))+((new_dict[1]['ind'])*(1+(new_dict[2]['ind'])/100))+((new_dict[1]['res'])*(
                        1+(new_dict[2]['res'])/100))+((new_dict[1]['pub'])*(1+(new_dict[2]['pub'])/100))+((new_dict[1]['agr'])*(1+(new_dict[2]['agr'])/100))+((new_dict[1]['other'])*(1+(new_dict[2]['other'])/100)))
                    selected_range_3.columns = selected_range_3.columns + \
                        relativedelta(years=1)
                    selected_range_4 = (selected_range_3/100)*(((new_dict[1]['com'])*(1+(new_dict[2]['com'])/100))+((new_dict[1]['ind'])*(1+(new_dict[2]['ind'])/100))+((new_dict[1]['res'])*(
                        1+(new_dict[2]['res'])/100))+((new_dict[1]['pub'])*(1+(new_dict[2]['pub'])/100))+((new_dict[1]['agr'])*(1+(new_dict[2]['agr'])/100))+((new_dict[1]['other'])*(1+(new_dict[2]['other'])/100)))
                    selected_range_4.columns = selected_range_4.columns + \
                        relativedelta(years=1)
                    selected_range_5 = (selected_range_4/100)*(((new_dict[1]['com'])*(1+(new_dict[2]['com'])/100))+((new_dict[1]['ind'])*(1+(new_dict[2]['ind'])/100))+((new_dict[1]['res'])*(
                        1+(new_dict[2]['res'])/100))+((new_dict[1]['pub'])*(1+(new_dict[2]['pub'])/100))+((new_dict[1]['agr'])*(1+(new_dict[2]['agr'])/100))+((new_dict[1]['other'])*(1+(new_dict[2]['other'])/100)))
                    selected_range_5.columns = selected_range_5.columns + \
                        relativedelta(years=1)

                    required_range = pd.concat(
                        [selected_range, Transformer_capacity], axis=1, sort=False).T.to_numpy()
                    required_range_2 = pd.concat(
                        [selected_range_2, Transformer_capacity], axis=1, sort=False).T.to_numpy()
                    required_range_3 = pd.concat(
                        [selected_range_3, Transformer_capacity], axis=1, sort=False).T.to_numpy()
                    required_range_4 = pd.concat(
                        [selected_range_4, Transformer_capacity], axis=1, sort=False).T.to_numpy()
                    required_range_5 = pd.concat(
                        [selected_range_5, Transformer_capacity], axis=1, sort=False).T.to_numpy()

                    required_range = required_range.reshape(
                        (selected_range.shape[1]+2, int(1440/resolution)))
                    required_axis = numpy.arange(int(1440/resolution))
                    segs = numpy.zeros(
                        (selected_range.shape[1]+2, int(1440/resolution), 2))
                    segs[:, :, 1] = required_range
                    segs[:, :, 0] = required_axis
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(required_axis.min(), required_axis.max())
                    ax.set_ylim(required_range.min(), required_range.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments = LineCollection(segs, linewidths=(0.5),
                                                   colors=colors, linestyle='solid')
                    ax.add_collection(line_segments)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load (kW) of the given DT \nfor chosen period in Year 1", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks['Base load'] = f'media/outputs/{folderId}/Base load.png'

                    required_range_2 = required_range_2.reshape(
                        (selected_range_2.shape[1]+2, int(1440/resolution)))
                    required_axis_2 = numpy.arange(int(1440/resolution))
                    segs_2 = numpy.zeros(
                        (selected_range_2.shape[1]+2, int(1440/resolution), 2))
                    segs_2[:, :, 1] = required_range_2
                    segs_2[:, :, 0] = required_axis_2
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(required_axis_2.min(), required_axis_2.max())
                    ax.set_ylim(required_range_2.min(), required_range_2.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_2 = LineCollection(segs_2, linewidths=(0.5),
                                                     colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_2)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load (kW) of the given DT \nfor chosen period in Year 2", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load_2',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks['Base load_2'] = f'media/outputs/{folderId}/Base load_2.png'
                    required_range_3 = required_range_3.reshape(
                        (selected_range_3.shape[1]+2, int(1440/resolution)))
                    required_axis_3 = numpy.arange(int(1440/resolution))
                    segs_3 = numpy.zeros(
                        (selected_range_3.shape[1]+2, int(1440/resolution), 2))
                    segs_3[:, :, 1] = required_range_3
                    segs_3[:, :, 0] = required_axis_3
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(required_axis_3.min(), required_axis_3.max())
                    ax.set_ylim(required_range_3.min(), required_range_3.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_3 = LineCollection(segs_3, linewidths=(0.5),
                                                     colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_3)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load (kW) of the given DT \nfor chosen period in Year 3", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load_3',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks['Base load_3'] = f'media/outputs/{folderId}/Base load_3.png'

                    required_range_4 = required_range_4.reshape(
                        (selected_range_4.shape[1]+2, int(1440/resolution)))
                    required_axis_4 = numpy.arange(int(1440/resolution))
                    segs_4 = numpy.zeros(
                        (selected_range_4.shape[1]+2, int(1440/resolution), 2))
                    segs_4[:, :, 1] = required_range_4
                    segs_4[:, :, 0] = required_axis_4
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(required_axis_4.min(), required_axis_4.max())
                    ax.set_ylim(required_range_4.min(), required_range_4.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_4 = LineCollection(segs_4, linewidths=(0.5),
                                                     colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_4)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load (kW) of the given DT \nfor chosen period in Year 4", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load_4',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks['Base load_4'] = f'media/outputs/{folderId}/Base load_4.png'

                    required_range_5 = required_range_5.reshape(
                        (selected_range_5.shape[1]+2, int(1440/resolution)))
                    required_axis_5 = numpy.arange(int(1440/resolution))
                    segs_5 = numpy.zeros(
                        (selected_range_5.shape[1]+2, int(1440/resolution), 2))
                    segs_5[:, :, 1] = required_range_5
                    segs_5[:, :, 0] = required_axis_5
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(required_axis_5.min(), required_axis_5.max())
                    ax.set_ylim(required_range_5.min(), required_range_5.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_5 = LineCollection(segs_5, linewidths=(0.5),
                                                     colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_5)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load (kW) of the given DT \nfor chosen period in Year 5", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load_5',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks['Base load_5'] = f'media/outputs/{folderId}/Base load_5.png'

                    shazam = pd.DataFrame(ddlj)

                    # -- cunstome --
                    # if numOfvehicleCategory == 1:
                    #     ddf2 = ddf.mul([((my_dict[1]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_2 = pd.DataFrame((ddf2.sum(axis=0)).to_numpy())
                    #     ddf3 = ddf2.mul([((my_dict[1]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_3 = pd.DataFrame((ddf3.sum(axis=0)).to_numpy())
                    #     ddf4 = ddf3.mul([((my_dict[1]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_4 = pd.DataFrame((ddf4.sum(axis=0)).to_numpy())
                    #     ddf5 = ddf4.mul([((my_dict[1]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_5 = pd.DataFrame((ddf5.sum(axis=0)).to_numpy())

                    # if numOfvehicleCategory == 2:
                    #     ddf2 = ddf.mul(
                    #         [((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_2 = pd.DataFrame((ddf2.sum(axis=0)).to_numpy())
                    #     ddf3 = ddf2.mul(
                    #         [((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_3 = pd.DataFrame((ddf3.sum(axis=0)).to_numpy())
                    #     ddf4 = ddf3.mul(
                    #         [((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_4 = pd.DataFrame((ddf4.sum(axis=0)).to_numpy())
                    #     ddf5 = ddf4.mul(
                    #         [((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_5 = pd.DataFrame((ddf5.sum(axis=0)).to_numpy())

                    # if numOfvehicleCategory == 3:
                    #     ddf2 = ddf.mul([((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1,
                    #                     ((my_dict[3]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_2 = pd.DataFrame((ddf2.sum(axis=0)).to_numpy())
                    #     ddf3 = ddf2.mul([((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1,
                    #                     ((my_dict[3]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_3 = pd.DataFrame((ddf3.sum(axis=0)).to_numpy())
                    #     ddf4 = ddf3.mul([((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1,
                    #                     ((my_dict[3]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_4 = pd.DataFrame((ddf4.sum(axis=0)).to_numpy())
                    #     ddf5 = ddf4.mul([((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1,
                    #                     ((my_dict[3]['CAGR_V'])/100)+1], axis=0)
                    #     shazam_5 = pd.DataFrame((ddf5.sum(axis=0)).to_numpy())

                    # if numOfvehicleCategory == 4:
                    ddf2 = ddf.mul([((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1,
                                    ((my_dict[3]['CAGR_V'])/100)+1, ((my_dict[4]['CAGR_V'])/100)+1], axis=0)
                    shazam_2 = pd.DataFrame((ddf2.sum(axis=0)).to_numpy())
                    ddf3 = ddf2.mul([((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1,
                                    ((my_dict[3]['CAGR_V'])/100)+1, ((my_dict[4]['CAGR_V'])/100)+1], axis=0)
                    shazam_3 = pd.DataFrame((ddf3.sum(axis=0)).to_numpy())
                    ddf4 = ddf3.mul([((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1,
                                    ((my_dict[3]['CAGR_V'])/100)+1, ((my_dict[4]['CAGR_V'])/100)+1], axis=0)
                    shazam_4 = pd.DataFrame((ddf4.sum(axis=0)).to_numpy())
                    ddf5 = ddf4.mul([((my_dict[1]['CAGR_V'])/100)+1, ((my_dict[2]['CAGR_V'])/100)+1,
                                    ((my_dict[3]['CAGR_V'])/100)+1, ((my_dict[4]['CAGR_V'])/100)+1], axis=0)
                    shazam_5 = pd.DataFrame((ddf5.sum(axis=0)).to_numpy())
                    # else:
                    #     print("will just go with 4 vehicleCategory input fields")
                    # -- custome --

                    required_net_load = pd.DataFrame(
                        selected_range) + shazam.values
                    required_net_load_2 = pd.DataFrame(
                        selected_range_2) + shazam_2.values
                    required_net_load_3 = pd.DataFrame(
                        selected_range_3) + shazam_3.values
                    required_net_load_4 = pd.DataFrame(
                        selected_range_4) + shazam_4.values
                    required_net_load_5 = pd.DataFrame(
                        selected_range_5) + shazam_5.values

                    net_load_curve = pd.concat(
                        [required_net_load, Transformer_capacity], axis=1, sort=False).T.to_numpy()
                    net_load_curve_2 = pd.concat(
                        [required_net_load_2, Transformer_capacity], axis=1, sort=False).T.to_numpy()
                    net_load_curve_3 = pd.concat(
                        [required_net_load_3, Transformer_capacity], axis=1, sort=False).T.to_numpy()
                    net_load_curve_4 = pd.concat(
                        [required_net_load_4, Transformer_capacity], axis=1, sort=False).T.to_numpy()
                    net_load_curve_5 = pd.concat(
                        [required_net_load_5, Transformer_capacity], axis=1, sort=False).T.to_numpy()

                    net_load_curve = net_load_curve.reshape(
                        (selected_range.shape[1]+2, int(1440/resolution)))
                    net_load_required_axis = numpy.arange(int(1440/resolution))
                    segs_n = numpy.zeros(
                        (selected_range.shape[1]+2, int(1440/resolution), 2))
                    segs_n[:, :, 1] = net_load_curve
                    segs_n[:, :, 0] = net_load_required_axis
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(net_load_required_axis.min(),
                                net_load_required_axis.max())
                    ax.set_ylim(net_load_curve.min(), net_load_curve.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_n = LineCollection(segs_n, linewidths=(0.5),
                                                     colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_n)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load + EV load (kW) of the given DT \nfor chosen period in Year 1", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load + EV load (kW) in Year 1',
                                bbox_inches='tight', dpi=300)
                    plt.clf()

                    outputLinks[
                        'Base load + EV load (kW) in Year 1'] = f'media/outputs/{folderId}/Base load + EV load (kW) in Year 1.png'

                    net_load_curve_2 = net_load_curve_2.reshape(
                        (selected_range_2.shape[1]+2, int(1440/resolution)))
                    net_load_required_axis_2 = numpy.arange(
                        int(1440/resolution))
                    segs_n_2 = numpy.zeros(
                        (selected_range_2.shape[1]+2, int(1440/resolution), 2))
                    segs_n_2[:, :, 1] = net_load_curve_2
                    segs_n_2[:, :, 0] = net_load_required_axis_2
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(net_load_required_axis_2.min(),
                                net_load_required_axis_2.max())
                    ax.set_ylim(net_load_curve_2.min(), net_load_curve_2.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_n_2 = LineCollection(segs_n_2, linewidths=(0.5),
                                                       colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_n_2)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load + EV load (kW) of the given DT \nfor chosen period in Year 2", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load + EV load (kW) in Year 2',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks[
                        'Base load + EV load (kW) in Year 2'] = f'media/outputs/{folderId}/Base load + EV load (kW) in Year 2.png'
                    net_load_curve_3 = net_load_curve_3.reshape(
                        (selected_range_3.shape[1]+2, int(1440/resolution)))
                    net_load_required_axis_3 = numpy.arange(
                        int(1440/resolution))
                    segs_n_3 = numpy.zeros(
                        (selected_range_3.shape[1]+2, int(1440/resolution), 2))
                    segs_n_3[:, :, 1] = net_load_curve_3
                    segs_n_3[:, :, 0] = net_load_required_axis_3
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(net_load_required_axis_3.min(),
                                net_load_required_axis_3.max())
                    ax.set_ylim(net_load_curve_3.min(), net_load_curve_3.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_n_3 = LineCollection(segs_n_3, linewidths=(0.5),
                                                       colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_n_3)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load + EV load (kW) of the given DT \nfor chosen period in Year 3", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load + EV load (kW) in Year 3',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks[
                        'Base load + EV load (kW) in Year 3'] = f'media/outputs/{folderId}/Base load + EV load (kW) in Year 3.png'
                    net_load_curve_4 = net_load_curve_4.reshape(
                        (selected_range_4.shape[1]+2, int(1440/resolution)))
                    net_load_required_axis_4 = numpy.arange(
                        int(1440/resolution))
                    segs_n_4 = numpy.zeros(
                        (selected_range_4.shape[1]+2, int(1440/resolution), 2))
                    segs_n_4[:, :, 1] = net_load_curve_4
                    segs_n_4[:, :, 0] = net_load_required_axis_4
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(net_load_required_axis_4.min(),
                                net_load_required_axis_4.max())
                    ax.set_ylim(net_load_curve_4.min(), net_load_curve_4.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_n_4 = LineCollection(segs_n_4, linewidths=(0.5),
                                                       colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_n_4)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load + EV load (kW) of the given DT \nfor chosen period in Year 4", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load + EV load (kW) in Year 4',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks[
                        'Base load + EV load (kW) in Year 4'] = f'media/outputs/{folderId}/Base load + EV load (kW) in Year 4.png'
                    net_load_curve_5 = net_load_curve_5.reshape(
                        (selected_range_5.shape[1]+2, int(1440/resolution)))
                    net_load_required_axis_5 = numpy.arange(
                        int(1440/resolution))
                    segs_n_5 = numpy.zeros(
                        (selected_range_5.shape[1]+2, int(1440/resolution), 2))
                    segs_n_5[:, :, 1] = net_load_curve_5
                    segs_n_5[:, :, 0] = net_load_required_axis_5
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(net_load_required_axis_5.min(),
                                net_load_required_axis_5.max())
                    ax.set_ylim(net_load_curve_5.min(), net_load_curve_5.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments_n_5 = LineCollection(segs_n_5, linewidths=(0.5),
                                                       colors=colors, linestyle='solid')
                    ax.add_collection(line_segments_n_5)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load + EV load (kW) of the given DT \nfor chosen period in Year 5", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load + EV load (kW) in Year 5',
                                bbox_inches='tight', dpi=300)
                    plt.clf()
                    outputLinks[
                        'Base load + EV load (kW) in Year 5'] = f'media/outputs/{folderId}/Base load + EV load (kW) in Year 5.png'
                    # Density mapping of potential transformer capacity breaches
                    overshot = required_net_load - \
                        (excelimport.iloc[0, 1]*(BR_F)/100)
                    overshot_2 = required_net_load_2 - \
                        (excelimport.iloc[0, 1]*(BR_F)/100)
                    overshot_3 = required_net_load_3 - \
                        (excelimport.iloc[0, 1]*(BR_F)/100)
                    overshot_4 = required_net_load_4 - \
                        (excelimport.iloc[0, 1]*(BR_F)/100)
                    overshot_5 = required_net_load_5 - \
                        (excelimport.iloc[0, 1]*(BR_F)/100)

                    overshot_r = required_net_load - \
                        excelimport.iloc[0, 1]*(90/100)
                    overshot_2_r = required_net_load_2 - \
                        excelimport.iloc[0, 1]*(90/100)
                    overshot_3_r = required_net_load_3 - \
                        excelimport.iloc[0, 1]*(90/100)
                    overshot_4_r = required_net_load_4 - \
                        excelimport.iloc[0, 1]*(90/100)
                    overshot_5_r = required_net_load_5 - \
                        excelimport.iloc[0, 1]*(90/100)

                    overshot[overshot < 0] = 0
                    overshot_2[overshot_2 < 0] = 0
                    overshot_3[overshot_3 < 0] = 0
                    overshot_4[overshot_4 < 0] = 0
                    overshot_5[overshot_5 < 0] = 0

                    overshot_r[overshot_r < 0] = 0
                    overshot_2_r[overshot_2_r < 0] = 0
                    overshot_3_r[overshot_3_r < 0] = 0
                    overshot_4_r[overshot_4_r < 0] = 0
                    overshot_5_r[overshot_5_r < 0] = 0

                    # Data for the summary table
                    Table_summary = [
                        ['Max load excursion beyond the \ntrigger for planning (kW)', 'Number of such excursion in \na year (30 minute blocks)',
                         'Max load excursion beyond 100% of \nthe DT installed capacity (kW)', 'Number of such excursion in \na year (30 minute blocks)'],
                        ['Year 1', round(overshot.max().max(), 2), (overshot != 0).values.sum(), round(
                            overshot_r.max().max(), 2), (overshot_r != 0).values.sum()],
                        ['Year 2', round(overshot_2.max().max(), 2), (overshot_2 != 0).values.sum(
                        ), round(overshot_2_r.max().max(), 2), (overshot_2_r != 0).values.sum()],
                        ['Year 3', round(overshot_3.max().max(), 2), (overshot_3 != 0).values.sum(
                        ), round(overshot_3_r.max().max(), 2), (overshot_3_r != 0).values.sum()],
                        ['Year 4', round(overshot_4.max().max(), 2), (overshot_4 != 0).values.sum(
                        ), round(overshot_4_r.max().max(), 2), (overshot_4_r != 0).values.sum()],
                        ['Year 5', round(overshot_5.max().max(), 2), (overshot_5 != 0).values.sum(
                        ), round(overshot_5_r.max().max(), 2), (overshot_5_r != 0).values.sum()],
                    ]

                    # Pop the headers from the data array
                    column_headers = Table_summary.pop(0)
                    row_headers = [x.pop(0) for x in Table_summary]
                    # Table data needs to be non-numeric text. Format the data
                    cell_text = []
                    for row in Table_summary:
                        cell_text.append([f'{x}' for x in row])

                    # Get some lists of color specs for row and column headers
                    rcolors = plt.cm.BuPu(numpy.full(len(row_headers), 0.1))
                    ccolors = plt.cm.BuPu(numpy.full(len(column_headers), 0.1))
                    # Create the figure. Setting a small pad on tight_layout
                    # seems to better regulate white space. Sometimes experimenting
                    # with an explicit figsize here can produce better outcome.
                    plt.figure(linewidth=2,
                               # edgecolor='steelblue',
                               # facecolor='skyblue',
                               tight_layout={},
                               figsize=(16, 8),
                               dpi=600,
                               )

                    # Add a table at the bottom of the axes
                    the_table = plt.table(cellText=cell_text,
                                          fontsize=500,
                                          # wrap=True,
                                          rowLabels=row_headers,
                                          rowColours=rcolors,
                                          rowLoc='right',
                                          colColours=ccolors,
                                          colLabels=column_headers,
                                          colWidths=[0.1, 0.1, 0.1, 0.1],
                                          loc='center')

                    the_table.scale(2, 5)
                    # cellDict = the_table.get_celld()
                    # for i in range(0,len(column_headers)+1):
                    #    cellDict[(0,i)].set_height(0.4)
                    #    for j in range(1,len(Table_summary)+1):
                    #        cellDict[(j,i)].set_height(0.2)
                    # set_text_props(wrap=True)
                    # Scaling is the only influence we have over top and bottom cell padding.
                    # Make the rows taller (i.e., make cell y scale larger).
                    # the_table.scale(1, 1)
                    # Hide axes
                    ax = plt.gca()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    # Hide axes border
                    plt.box(on=None)
                    # Force the figure to update, so backends center objects correctly within the figure.
                    # Without plt.draw() here, the title will center on the axes and not the figure.
                    plt.draw()
                    # Create image. plt.savefig ignores figure edge and face colors, so map them.
                    fig = plt.gcf()
                    plt.savefig(f'media/outputs/{folderId}/summary_table.png',
                                # bbox='tight',
                                # edgecolor=fig.get_edgecolor(),
                                # facecolor=fig.get_facecolor(),
                                dpi=600
                                )
                    plt.clf()
                    outputLinks['summary_table.png'] = f'media/outputs/{folderId}/summary_table.png'
                    overshot.columns = overshot.columns.astype(str)
                    overshot_2.columns = overshot_2.columns.astype(str)
                    overshot_3.columns = overshot_3.columns.astype(str)
                    overshot_4.columns = overshot_4.columns.astype(str)
                    overshot_5.columns = overshot_5.columns.astype(str)

                    overshot_r.columns = overshot_r.columns.astype(str)
                    overshot_2_r.columns = overshot_2_r.columns.astype(str)
                    overshot_3_r.columns = overshot_3_r.columns.astype(str)
                    overshot_4_r.columns = overshot_4_r.columns.astype(str)
                    overshot_5_r.columns = overshot_5_r.columns.astype(str)

                    # solar_hours = numpy.arange(0, 48, 1).reshape(48,1)
                    # shm2 = 26.26 #1:08 PM peak of solar generation converted into 30 min blocks
                    # shms2 = 3
                    # solarfactor_1=overshot.values.sum()
                    # solarfactor_2=overshot_2.values.sum()
                    # solarfactor_3=overshot_3.values.sum()
                    # solarfactor_4=overshot_4.values.sum()
                    # solarfactor_5=overshot_5.values.sum()

                    # S_generation_1 = (ss.norm.pdf(solar_hours, shm2, shms2))*solarfactor_1
                    # solar_gen_1=pd.DataFrame(data = S_generation_1, columns=["Required solar generation"])

                    # S_generation_2 = (ss.norm.pdf(solar_hours, shm2, shms2))*solarfactor_2
                    # solar_gen_2=pd.DataFrame(data = S_generation_2, columns=["Required solar generation"])

                    # S_generation_3 = (ss.norm.pdf(solar_hours, shm2, shms2))*solarfactor_3
                    # solar_gen_3=pd.DataFrame(data = S_generation_3,  columns=["Required solar generation"])

                    # S_generation_4 = (ss.norm.pdf(solar_hours, shm2, shms2))*solarfactor_4
                    # solar_gen_4=pd.DataFrame(data = S_generation_4, columns=["Required solar generation"])

                    # S_generation_5 = (ss.norm.pdf(solar_hours, shm2, shms2))*solarfactor_5
                    # solar_gen_5=pd.DataFrame(data = S_generation_5,  columns=["Required solar generation"])

                    # cat=pd.concat([overshot, solar_gen_1], axis=1, sort=False)
                    overshot.reset_index(level=0, inplace=True)
                    overshot.rename(columns={'index': 'slot'}, inplace=True)
                    overshot_p = pd.melt(overshot, id_vars=["slot"])

                    overshot_r.reset_index(level=0, inplace=True)
                    overshot_r.rename(columns={'index': 'slot'}, inplace=True)
                    overshot_p_r = pd.melt(overshot_r, id_vars=["slot"])

                    # cat2=pd.concat([overshot_2, solar_gen_2], axis=1, sort=False)
                    overshot_2.reset_index(level=0, inplace=True)
                    overshot_2.rename(columns={'index': 'slot'}, inplace=True)
                    overshot_p_2 = pd.melt(overshot_2, id_vars=["slot"])

                    overshot_2_r.reset_index(level=0, inplace=True)
                    overshot_2_r.rename(
                        columns={'index': 'slot'}, inplace=True)
                    overshot_p_2_r = pd.melt(overshot_2_r, id_vars=["slot"])

                    # cat3=pd.concat([overshot_3, solar_gen_3], axis=1, sort=False)
                    overshot_3.reset_index(level=0, inplace=True)
                    overshot_3.rename(columns={'index': 'slot'}, inplace=True)
                    overshot_p_3 = pd.melt(overshot_3, id_vars=["slot"])

                    overshot_3_r.reset_index(level=0, inplace=True)
                    overshot_3_r.rename(
                        columns={'index': 'slot'}, inplace=True)
                    overshot_p_3_r = pd.melt(overshot_3_r, id_vars=["slot"])

                    # cat4=pd.concat([overshot_4, solar_gen_4], axis=1, sort=False)
                    overshot_4.reset_index(level=0, inplace=True)
                    overshot_4.rename(columns={'index': 'slot'}, inplace=True)
                    overshot_p_4 = pd.melt(overshot_4, id_vars=["slot"])

                    overshot_4_r.reset_index(level=0, inplace=True)
                    overshot_4_r.rename(
                        columns={'index': 'slot'}, inplace=True)
                    overshot_p_4_r = pd.melt(overshot_4_r, id_vars=["slot"])

                    # cat5=pd.concat([overshot_5, solar_gen_5], axis=1, sort=False)
                    overshot_5.reset_index(level=0, inplace=True)
                    overshot_5.rename(columns={'index': 'slot'}, inplace=True)
                    overshot_p_5 = pd.melt(overshot_5, id_vars=["slot"])

                    overshot_5_r.reset_index(level=0, inplace=True)
                    overshot_5_r.rename(
                        columns={'index': 'slot'}, inplace=True)
                    overshot_p_5_r = pd.melt(overshot_5_r, id_vars=["slot"])

                    overshot_p.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                               'variable': 'dates', 'value': 'overshot load (kW) beyond planning trigger in Year 1'}, inplace=True)
                    overshot_p_2.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                        'variable': 'dates', 'value': 'overshot load (kW) beyond planning trigger in Year 2'}, inplace=True)
                    overshot_p_3.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                        'variable': 'dates', 'value': 'overshot load (kW) beyond planning trigger in Year 3'}, inplace=True)
                    overshot_p_4.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                        'variable': 'dates', 'value': 'overshot load (kW) beyond planning trigger in Year 4'}, inplace=True)
                    overshot_p_5.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                        'variable': 'dates', 'value': 'overshot load (kW) beyond planning trigger in Year 5'}, inplace=True)

                    overshot_p_r.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                        'variable': 'dates', 'value': 'overshot load (kW) beyond rated DT capacity in Year 1'}, inplace=True)
                    overshot_p_2_r.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                                   'variable': 'dates', 'value': 'overshot load (kW) beyond rated DT capacity in Year 2'}, inplace=True)
                    overshot_p_3_r.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                                   'variable': 'dates', 'value': 'overshot load (kW) beyond rated DT capacity in Year 3'}, inplace=True)
                    overshot_p_4_r.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                                   'variable': 'dates', 'value': 'overshot load (kW) beyond rated DT capacity in Year 4'}, inplace=True)
                    overshot_p_5_r.rename(columns={'slot': str(int(1440/resolution))+" time blocks of "+str(resolution)+" minutes each per day",
                                                   'variable': 'dates', 'value': 'overshot load (kW) beyond rated DT capacity in Year 5'}, inplace=True)

                    xlwriter = pd.ExcelWriter(
                        f'media/outputs/{folderId}/overshot_1.xlsx')
                    outputLinks['overshot_1.xlsx'] = f'media/outputs/{folderId}/overshot_1.xlsx'
                    overshot_p.to_excel(xlwriter, sheet_name='Sheet1')
                    overshot_p_r.to_excel(xlwriter, sheet_name='Sheet2')
                    xlwriter.close()

                    xlwriter = pd.ExcelWriter(
                        f'media/outputs/{folderId}/overshot_2.xlsx')
                    outputLinks['overshot_2.xlsx'] = f'media/outputs/{folderId}/overshot_2.xlsx'
                    overshot_p_2.to_excel(xlwriter, sheet_name='Sheet1')
                    overshot_p_2_r.to_excel(xlwriter, sheet_name='Sheet2')
                    xlwriter.close()

                    xlwriter = pd.ExcelWriter(
                        f'media/outputs/{folderId}/overshot_3.xlsx')
                    outputLinks['overshot_3.xlsx'] = f'media/outputs/{folderId}/overshot_3.xlsx'
                    overshot_p_3.to_excel(xlwriter, sheet_name='Sheet1')
                    overshot_p_3_r.to_excel(xlwriter, sheet_name='Sheet2')
                    xlwriter.close()

                    xlwriter = pd.ExcelWriter(
                        f'media/outputs/{folderId}/overshot_4.xlsx')
                    outputLinks['overshot_4.xlsx'] = f'media/outputs/{folderId}/overshot_4.xlsx'
                    overshot_p_4.to_excel(xlwriter, sheet_name='Sheet1')
                    overshot_p_4_r.to_excel(xlwriter, sheet_name='Sheet2')
                    xlwriter.close()

                    xlwriter = pd.ExcelWriter(
                        f'media/outputs/{folderId}/overshot_5.xlsx')
                    outputLinks['overshot_5.xlsx'] = f'media/outputs/{folderId}/overshot_5.xlsx'
                    overshot_p_5.to_excel(xlwriter, sheet_name='Sheet1')
                    overshot_p_5_r.to_excel(xlwriter, sheet_name='Sheet2')
                    xlwriter.close()

                    sns.set_style('darkgrid')
                    slots_array = slots['slot_labels'].astype(
                        str).values.tolist()
                    # slots_array_b=numpy.array('[%s]' % ', '.join(map(str, slots_array)))

                    # print (slots_array)
                    chart1 = alt.Chart(overshot_p).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond planning trigger in Year 1:Q', stack=None),
                        color="dates:O")
                    # curve=alt.Chart(solar_gen_1).mark_line().encode(
                    #    y=alt.Y('solar_gen_1:Q'))
                    # overlay+curve.save('overshot.html', scale_factor=3.0)
                    chart1.save(
                        f'media/outputs/{folderId}/overshot_density.html', scale_factor=3.0)
                    outputLinks['overshot_density.html'] = f'media/outputs/{folderId}/overshot_density.html'
                    sns.set_style('darkgrid')
                    chart2 = alt.Chart(overshot_p_2).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond planning trigger in Year 2:Q', stack=None),
                        color="dates:O")
                    # curve=alt.Chart(solar_gen_1).mark_line().encode(
                    #    y=alt.Y('solar_gen_1'))
                    # overlay+curve
                    chart2.save(
                        f'media/outputs/{folderId}/overshot_density_2.html', scale_factor=3.0)
                    outputLinks['overshot_density_2.html'] = f'media/outputs/{folderId}/overshot_density_2.html'
                    sns.set_style('darkgrid')
                    chart3 = alt.Chart(overshot_p_3).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond planning trigger in Year 3:Q', stack=None),
                        color="dates:O")
                    chart3.save(
                        f'media/outputs/{folderId}/overshot_density_3.html', scale_factor=3.0)
                    outputLinks[
                        'overshot_density.html_3.html'] = f'media/outputs/{folderId}/overshot_density_3.html'
                    sns.set_style('darkgrid')
                    chart4 = alt.Chart(overshot_p_4).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond planning trigger in Year 4:Q', stack=None),
                        color="dates:O")
                    chart4.save(
                        f'media/outputs/{folderId}/overshot_density_4.html', scale_factor=3.0)
                    outputLinks[
                        'overshot_density.html_4.html'] = f'media/outputs/{folderId}/overshot_density_4.html'
                    sns.set_style('darkgrid')
                    chart5 = alt.Chart(overshot_p_5).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond planning trigger in Year 5:Q', stack=None),
                        color="dates:O")
                    chart5.save(
                        f'media/outputs/{folderId}/overshot_density_5.html', scale_factor=3.0)
                    outputLinks[
                        'overshot_density.html_5.html'] = f'media/outputs/{folderId}/overshot_density_5.html'
                    chart6 = alt.Chart(overshot_p_r).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond rated DT capacity in Year 1:Q', stack=None),
                        color="dates:O")
                    # curve=alt.Chart(solar_gen_1).mark_line().encode(
                    #    y=alt.Y('solar_gen_1:Q'))
                    # overlay+curve.save('overshot.html', scale_factor=3.0)
                    chart6.save(
                        f'media/outputs/{folderId}/overshot_density_r.html', scale_factor=3.0)
                    outputLinks['overshot_density_r.html'] = f'media/outputs/{folderId}/overshot_density_r.html'
                    sns.set_style('darkgrid')
                    chart7 = alt.Chart(overshot_p_2_r).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond rated DT capacity in Year 2:Q', stack=None),
                        color="dates:O")
                    # curve=alt.Chart(solar_gen_1).mark_line().encode(
                    #    y=alt.Y('solar_gen_1'))
                    # overlay+curve
                    chart7.save(
                        f'media/outputs/{folderId}/overshot_density_2_r.html', scale_factor=3.0)
                    outputLinks['overshot_density_2_r.html'] = f'media/outputs/{folderId}/overshot_density_2_r.html'
                    sns.set_style('darkgrid')
                    chart8 = alt.Chart(overshot_p_3_r).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond rated DT capacity in Year 3:Q', stack=None),
                        color="dates:O")
                    chart8.save(
                        f'media/outputs/{folderId}/overshot_density_3_r.html', scale_factor=3.0)
                    outputLinks['overshot_density_3_r.html'] = f'media/outputs/{folderId}/overshot_density_3_r.html'
                    sns.set_style('darkgrid')
                    chart9 = alt.Chart(overshot_p_4_r).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond rated DT capacity in Year 4:Q', stack=None),
                        color="dates:O")
                    chart9.save(
                        f'media/outputs/{folderId}/overshot_density_4.html', scale_factor=3.0)
                    outputLinks['overshot_density_4.html'] = f'media/outputs/{folderId}/overshot_density_4.html'
                    sns.set_style('darkgrid')
                    chart10 = alt.Chart(overshot_p_5_r).mark_bar(opacity=0.3).encode(
                        x=alt.X(str(int(1440/resolution))+" time blocks of "+str(resolution) +
                                " minutes each per day:O", axis=alt.Axis(values=slots_array, labelAngle=90)),
                        y=alt.Y(
                            'overshot load (kW) beyond rated DT capacity in Year 5:Q', stack=None),
                        color="dates:O")
                    chart10.save(
                        f'media/outputs/{folderId}/overshot_density_5_r.html', scale_factor=3.0)
                    outputLinks['overshot_density_5_r.html'] = f'media/outputs/{folderId}/overshot_density_5_r.html'
                    # NEW CODE FROM HERE
                    r_str = str(resolution)+"min"
                    date_on = pd.date_range(
                        '2019-05-01', periods=406*24*60/resolution, freq=r_str)
                    abc['Date'] = pd.DataFrame(date_on)
                    abc['MM_DD_Code'] = pd.DataFrame(date_on)
                    abc['MM_DD_Code'] = abc['MM_DD_Code'].dt.strftime('%m%d')
                    abc['MM_DD_Code'] = abc['MM_DD_Code'].astype('int64')
                    # abc.to_excel('melt.xlsx')

                    # importing EV loads from simulation
                    excelimport_2 = pd.read_excel(
                        f"media/outputs/{folderId}/Simulated_EV_Load.xlsx", header=None)
                    ddf = excelimport_2.iloc[1:, 1:].T
                    ddf.reset_index(inplace=True)
                    # creating tiles of EV loads
                    ddf_x = pd.DataFrame(numpy.tile(ddf, (406, 1)))
                    ddf_x.columns = range(ddf_x.shape[1])
                    ddf_x.drop(ddf_x.columns[[0]], axis=1, inplace=True)

                    # arranging tiles of EV loads next to all daily base loads
                    abc_ddf = pd.concat([abc, ddf_x], axis=1)
                    abc_ddf['EV'] = ddf_x.sum(axis=1)
                    abc_ddf['Total_Load'] = abc_ddf['EV'] + abc_ddf[0]
                    abc_ddf['DT'] = DT[0]
                    abc_ddf['Overshot'] = numpy.where(
                        (abc_ddf['Total_Load']-abc_ddf['DT']) <= 0, 0, abc_ddf['Total_Load']-abc_ddf['DT'])

                    overshot_split = ddf_x.mul(abc_ddf['Overshot'], axis=0)
                    overshot_split = overshot_split.div(abc_ddf['EV'], axis=0)
                    overshot_split[overshot_split < 0] = 0
                    overshot_split.reset_index(inplace=True)
                    overshot_split.drop(
                        overshot_split.columns[[0]], axis=1, inplace=True)
                    overshot_split.columns = overshot_split.columns.astype(str)
                    overshot_split = overshot_split.rename(
                        columns=lambda x: x+'_%')
                    # overshot_split.to_excel('overshot_split.xlsx')
                    abc_ddf = pd.concat([abc_ddf, overshot_split], axis=1)

                    Code = pd.date_range('2020-01-01', periods=366, freq='D')
                    Code_DF = pd.DataFrame(Code)
                    Code_DF.columns = ['MM_DD_Code']
                    Code_DF['MM_DD_Code'] = Code_DF['MM_DD_Code'].dt.strftime(
                        '%m%d')
                    Code_DF['MM_DD_Code'] = Code_DF['MM_DD_Code'].astype(
                        'int64')

                    seasons['start_date'] = pd.to_datetime(
                        seasons['start_date'], format="%b-%d")
                    seasons['end_date'] = pd.to_datetime(
                        seasons['end_date'], format="%b-%d")
                    seasons['start_date'] = seasons['start_date'].dt.strftime(
                        '%m%d')
                    seasons['end_date'] = seasons['end_date'].dt.strftime(
                        '%m%d')
                    seasons_range = seasons.astype('int64')

                    conditions = [
                        (((seasons_range.iloc[0, 0] == abc['MM_DD_Code']) | (seasons_range.iloc[0, 0] < abc['MM_DD_Code'])) & (
                            (abc['MM_DD_Code'] == seasons_range.iloc[0, 1]) | (abc['MM_DD_Code'] < seasons_range.iloc[0, 1]))),
                        ((((seasons_range.iloc[1, 0] == abc['MM_DD_Code']) | (seasons_range.iloc[1, 0] < abc['MM_DD_Code'])) & ((abc['MM_DD_Code'] == 1231) | (abc['MM_DD_Code'] < 1231))) | (
                            (((seasons_range.iloc[1, 1] == abc['MM_DD_Code']) | (seasons_range.iloc[1, 1] > abc['MM_DD_Code'])) & (abc['MM_DD_Code'] > 0)))),
                        ((seasons_range.iloc[0, 0] != abc['MM_DD_Code']) & (seasons_range.iloc[0, 1] != abc['MM_DD_Code']) & (
                            seasons_range.iloc[1, 0] != abc['MM_DD_Code']) & (seasons_range.iloc[1, 1] != abc['MM_DD_Code']))
                    ]
                    values = ['summer', 'winter', 'offseason']
                    abc_ddf['seasons'] = numpy.select(conditions, values)
                    abc_ddf['hours'] = abc_ddf['Date'].dt.strftime('%H:%M')

                    def TOD_key(pks, pke, sx, ops, ope, rb):

                        pke = pke.replace("24:00", "23:59")
                        ope = ope.replace("24:00", "23:59")
                        FMT = '%H:%M'
                        pks = datetime.strptime(str(pks), FMT)
                        pke = datetime.strptime(str(pke), FMT)
                        ops = datetime.strptime(str(ops), FMT)
                        ope = datetime.strptime(str(ope), FMT)
                        TOD_Duration = pd.DataFrame({'peak_hours': [str(pke-pks)],
                                                    'offpeak_hours': [str(ope-ops)]})
                        return TOD_Duration

                    TOD_m = list()
                    for i, j in TOD.items():
                        res = TOD_key(**j)
                        TOD_m.append(res)
                    TOD_matrix = pd.concat(TOD_m)

                    def blast(x): return pd.Series(
                        [i for i in reversed(x.split(','))])

                    TOD_Blast_1 = TOD_matrix["peak_hours"].apply(blast)
                    TOD_Blast_2 = TOD_matrix["offpeak_hours"].apply(blast)
                    TOD_Fin = pd.concat(
                        [TOD_Blast_1, TOD_Blast_2], axis=1, join="inner")
                    TOD_Fin.loc[:, ~(TOD_Fin == 'day').any()]
                    TOD_Fin = TOD_Fin.dropna(
                        axis='columns').reset_index(drop=True)
                    TOD_Fin.columns = ['peak_hours', 'offpeak_hours']
                    TOD_Fin['peak_hours'] = TOD_Fin.apply(
                        lambda x: x['peak_hours'][:-3], axis=1)
                    TOD_Fin['offpeak_hours'] = TOD_Fin.apply(
                        lambda x: x['offpeak_hours'][:-3], axis=1)

                    def unit_blast(x): return pd.Series(
                        [i for i in x.split(':')])

                    TOD_unit_blast_1 = TOD_Fin["peak_hours"].apply(unit_blast)
                    TOD_unit_blast_2 = TOD_Fin["offpeak_hours"].apply(
                        unit_blast)
                    TOD_units = pd.concat(
                        [TOD_unit_blast_1, TOD_unit_blast_2], axis=1, join="inner")
                    TOD_units.columns = ['peak_hours', 'peak_minutes',
                                         'offpeak_hours', 'offpeak_minutes']
                    TOD_units = TOD_units.apply(pd.to_numeric)
                    TOD_units['peak_units'] = (
                        (TOD_units['peak_hours']*(60/resolution))+(TOD_units['peak_minutes']/resolution))
                    TOD_units['offpeak_units'] = (
                        (TOD_units['offpeak_hours']*(60/resolution))+(TOD_units['offpeak_minutes']/resolution))
                    TOD_units['peak_units'] = TOD_units['peak_units'].round(
                        0).astype('int64')
                    TOD_units['offpeak_units'] = TOD_units['offpeak_units'].round(
                        0).astype('int64')

                    abc_ddf = abc_ddf.set_index('Date')

                    TOD_df = pd.DataFrame(TOD)
                    TOD_df = TOD_df.T
                    TOD_df['pke'].loc[(TOD_df['pke'] == "24:00")] = "23:59"

                    conditions_2 = [
                        ((abc_ddf['seasons'] == 'summer') & (TOD_df.iloc[0, 0] < TOD_df.iloc[0, 1]) & (abc_ddf.index.isin(
                            abc_ddf.between_time(TOD_df.iloc[0, 0], TOD_df.iloc[0, 1], include_start=True, include_end=False).index))),
                        ((abc_ddf['seasons'] == 'summer') & (TOD_df.iloc[0, 0] > TOD_df.iloc[0, 1]) & ((abc_ddf.index.isin(abc_ddf.between_time(TOD_df.iloc[0, 0], "23:59", include_start=True,
                                                                                                                                                include_end=False).index)) | (abc_ddf.index.isin(abc_ddf.between_time("00:00", TOD_df.iloc[0, 1], include_start=True, include_end=False).index)))),
                        ((abc_ddf['seasons'] == 'summer') & (TOD_df.iloc[0, 3] < TOD_df.iloc[0, 4]) & (abc_ddf.index.isin(
                            abc_ddf.between_time(TOD_df.iloc[0, 3], TOD_df.iloc[0, 4], include_start=True, include_end=False).index))),
                        ((abc_ddf['seasons'] == 'summer') & (TOD_df.iloc[0, 3] > TOD_df.iloc[0, 4]) & ((abc_ddf.index.isin(abc_ddf.between_time(TOD_df.iloc[0, 3], "23:59", include_start=True,
                                                                                                                                                include_end=False).index)) | (abc_ddf.index.isin(abc_ddf.between_time("00:00", TOD_df.iloc[0, 4], include_start=True, include_end=False).index)))),
                        ((abc_ddf['seasons'] == 'winter') & (TOD_df.iloc[1, 0] < TOD_df.iloc[1, 1]) & (abc_ddf.index.isin(
                            abc_ddf.between_time(TOD_df.iloc[1, 0], TOD_df.iloc[1, 1], include_start=True, include_end=False).index))),
                        ((abc_ddf['seasons'] == 'winter') & (TOD_df.iloc[1, 0] > TOD_df.iloc[1, 1]) & ((abc_ddf.index.isin(abc_ddf.between_time(TOD_df.iloc[1, 0], "23:59", include_start=True,
                                                                                                                                                include_end=False).index)) | (abc_ddf.index.isin(abc_ddf.between_time("00:00", TOD_df.iloc[1, 1], include_start=True, include_end=False).index)))),
                        ((abc_ddf['seasons'] == 'winter') & (TOD_df.iloc[1, 3] < TOD_df.iloc[1, 4]) & (abc_ddf.index.isin(
                            abc_ddf.between_time(TOD_df.iloc[1, 3], TOD_df.iloc[1, 4], include_start=True, include_end=False).index))),
                        ((abc_ddf['seasons'] == 'winter') & (TOD_df.iloc[1, 3] > TOD_df.iloc[1, 4]) & ((abc_ddf.index.isin(abc_ddf.between_time(TOD_df.iloc[1, 3], "23:59",
                                                                                                                                                include_start=True, include_end=False).index)) | (abc_ddf.index.isin(abc_ddf.between_time("00:00", TOD_df.iloc[1, 4], include_start=True, include_end=False).index))))
                    ]

                    values_2 = ['pk', 'pk', 'op', 'op', 'pk', 'pk', 'op', 'op']
                    abc_ddf['unit_type'] = numpy.select(conditions_2, values_2)

                    overshot_split_1 = overshot_split
                    overshot_split_1['unit_type'] = abc_ddf['unit_type'].values
                    overshot_split_1.index = abc_ddf.index
                    overshot_split_1.loc[overshot_split_1['unit_type'] != "pk"] = 0
                    overshot_split_1 = overshot_split_1.drop(
                        ['unit_type'], axis=1)
                    deltas = overshot_split_1.groupby(
                        overshot_split_1.index.date).transform('sum')
                    deltas['unit_type'] = abc_ddf['unit_type'].values
                    deltas.loc[deltas['unit_type'] != "op"] = 0
                    deltas = deltas.drop(['unit_type'], axis=1)
                    deltas['unit_type'] = abc_ddf['unit_type'].values
                    deltas['seasons'] = abc_ddf['seasons'].values

                    conditions_3 = [
                        ((deltas['seasons'] == 'summer')
                         & (deltas['unit_type'] == 'pk')),
                        ((deltas['seasons'] == 'winter')
                         & (deltas['unit_type'] == 'pk')),
                        ((deltas['seasons'] == 'summer')
                         & (deltas['unit_type'] == 'op')),
                        ((deltas['seasons'] == 'winter')
                         & (deltas['unit_type'] == 'op')),
                        (deltas['unit_type'] == '0')
                    ]

                    values_3 = [TOD_units.iloc[0, 4], TOD_units.iloc[1, 4],
                                TOD_units.iloc[0, 5], TOD_units.iloc[1, 5], 0]
                    deltas['unit_div'] = numpy.select(conditions_3, values_3)
                    deltas = deltas.drop(['unit_type', 'seasons'], axis=1)
                    deltas = deltas.iloc[:, 0:-
                                         1].div(deltas['unit_div'], axis=0)
                    deltas.columns = ddf_x.columns
                    deltas.reset_index(inplace=True)
                    deltas = deltas.drop(['Date'], axis=1)
                    # deltas.to_excel('overshot_split_2.xlsx')

                    overshot_split_3 = overshot_split
                    overshot_split_3['unit_type'] = abc_ddf['unit_type'].values
                    overshot_split_3.loc[overshot_split_3['unit_type'] == "op"] = 0
                    overshot_split_3 = overshot_split_3.drop(
                        ['unit_type'], axis=1)
                    overshot_split_3.columns = ddf_x.columns
                    overshot_split_3.reset_index(inplace=True)
                    overshot_split_3 = overshot_split_3.drop(['Date'], axis=1)

                    ddf_x.set_index = overshot_split_3.index

                    inter_res = ddf_x.sub(
                        overshot_split_3, fill_value=0, axis=0)
                    final_res = inter_res.add(deltas, fill_value=0, axis=0)
                    # final_res.set_index=abc_ddf.index
                    final_res.index = abc_ddf.index
                    final_res.columns = final_res.columns.astype(str)
                    final_res = final_res.rename(columns=lambda x: x+'_New')

                    final_res['EV_New'] = final_res.sum(axis=1)

                    # final_res.to_excel('prefinal_3.xlsx')

                    # abc_ddf.to_excel('abc_ddf_1.xlsx')
                    # inter_res.to_excel('abc_ddf_2.xlsx')

                    # this step appends total new load EV curves to the results sheet
                    abc_ddf = pd.concat([abc_ddf, final_res], axis=1)
                    abc_ddf['Total_Load_New'] = abc_ddf[0] + abc_ddf['EV_New']

                    final_res_res = pd.DataFrame(numpy.reshape(
                        abc_ddf['Total_Load_New'].to_numpy(), (-1, int(1440/resolution)))).T
                    final_res_res.columns = labels
                    # final_res_res.to_excel('abc_ddf_v.xlsx')

                    final_res_res_1 = final_res_res
                    final_res_res_1['DT'] = numpy.tile(
                        excelimport.iloc[0, 1], (int(1440/resolution)))

                    final_res_res_1 = final_res_res.T.to_numpy()

                    required_axis = numpy.arange(int(1440/resolution))
                    segs = numpy.zeros(
                        (final_res_res.shape[1], int(1440/resolution), 2))
                    segs[:, :, 1] = final_res_res_1
                    segs[:, :, 0] = required_axis
                    plt.figure(figsize=(10, 15))
                    fig, ax = plt.subplots()
                    ax.set_xlim(required_axis.min(), required_axis.max())
                    ax.set_ylim(final_res_res_1.min(), final_res_res_1.max())
                    colors = [mcolors.to_rgba(c)
                              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
                    line_segments = LineCollection(segs, linewidths=(0.5),
                                                   colors=colors, linestyle='solid')
                    ax.add_collection(line_segments)
                    plt.xlabel(str(int(1440/resolution))+" time blocks of " +
                               str(resolution)+" minutes each per day", fontsize=12)
                    plt.ylabel(
                        "Base load + ToD EV load (kW) of the given DT \nin response to ToD tariff regime \nfor chosen period in Year 1", fontsize=12)
                    plt.savefig(f'media/outputs/{folderId}/Base load + ToD EV load',
                                bbox_inches='tight', dpi=300)
                    outputLinks['Base load + ToD EV load.png'] = f'media/outputs/{folderId}/Base load + ToD EV load.png'
                    conditions_4 = [
                        ((abc_ddf['seasons'] == 'summer')
                         & (abc_ddf['unit_type'] == 'pk')),
                        ((abc_ddf['seasons'] == 'winter')
                         & (abc_ddf['unit_type'] == 'pk')),
                        ((abc_ddf['seasons'] == 'summer')
                         & (abc_ddf['unit_type'] == 'op')),
                        ((abc_ddf['seasons'] == 'winter')
                         & (abc_ddf['unit_type'] == 'op')),
                        ((abc_ddf['seasons'] == 'summer')
                         & (abc_ddf['unit_type'] == '0')),
                        ((abc_ddf['seasons'] == 'winter')
                         & (abc_ddf['unit_type'] == '0')),
                    ]

                    values_4 = [sum_pk_cost, win_pk_cost, sum_op_cost,
                                win_op_cost, sum_0_cost, win_0_cost]
                    abc_ddf['utility_proc_tariff'] = numpy.select(
                        conditions_4, values_4)

                    ddf_x.index = abc_ddf.index

                    old_utility_cost = (
                        ddf_x.mul(abc_ddf['utility_proc_tariff'], axis=0)/2)
                    old_utility_cost.index = abc_ddf.index
                    old_utility_cost.columns = ddf_x.columns.astype(str)
                    old_utility_cost = old_utility_cost.rename(
                        columns=lambda x: x+'_old_cost')
                    # old_utility_cost.to_excel('old_utility_cost.xlsx')

                    new_utility_cost = (final_res.loc[:, final_res.columns != 'EV_New'].mul(
                        abc_ddf['utility_proc_tariff'], axis=0)/2)
                    new_utility_cost.index = abc_ddf.index
                    new_utility_cost.columns = ddf_x.columns.astype(str)
                    new_utility_cost = new_utility_cost.rename(
                        columns=lambda x: x+'_new_cost')
                    # new_utility_cost.to_excel('new_utility_cost.xlsx')

                    old_tariff_revenue = (
                        (ddf_x.mul(numpy.array(retail_tariff_df), axis=1))/2)
                    old_tariff_revenue.index = abc_ddf.index
                    old_tariff_revenue.columns = ddf_x.columns.astype(str)
                    old_tariff_revenue = old_tariff_revenue.rename(
                        columns=lambda x: x+'_old_tariff')
                    # old_tariff_revenue.to_excel('old_tariff_revenue.xlsx')

                    # new_tariff_revenue = ((final_res.loc[: , final_res.columns != 'EV_New'].mul(numpy.array(retail_tariff_df), axis=1))/2)
                    # new_tariff_revenue.index=abc_ddf.index
                    # new_tariff_revenue.columns=ddf_x.columns.astype(str)
                    # new_tariff_revenue = new_tariff_revenue.rename(columns=lambda x: x+'_new_tariff')
                    # new_tariff_revenue.to_excel('new_tariff_revenue.xlsx')

                    abc_ddf = pd.concat(
                        [abc_ddf, old_utility_cost, new_utility_cost, old_tariff_revenue], axis=1)
                    # abc_ddf= pd.concat([abc_ddf, old_utility_cost, new_utility_cost, old_tariff_revenue, new_tariff_revenue], axis=1)
                    abc_ddf.to_excel(
                        f'media/outputs/{folderId}/Load_Simulation_ToD_Calculation_Data.xlsx')
                    outputLinks[
                        'Load_Simulation_ToD_Calculation_Data.xls'] = f'media/outputs/{folderId}/Load_Simulation_ToD_Calculation_Data.xls'
                    abc_ddf.columns = abc_ddf.columns.astype(str)

                    s_pk_sum_1 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == 'pk'), '1'].sum()
                    s_op_sum_1 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == 'op'), '1'].sum()
                    s_0_sum_1 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == '0'), '1'].sum()
                    w_pk_sum_1 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == 'pk'), '1'].sum()
                    w_op_sum_1 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == 'op'), '1'].sum()
                    w_0_sum_1 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == '0'), '1'].sum()

                    s_pk_sum_2 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == 'pk'), '2'].sum()
                    s_op_sum_2 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == 'op'), '2'].sum()
                    s_0_sum_2 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == '0'), '2'].sum()
                    w_pk_sum_2 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == 'pk'), '2'].sum()
                    w_op_sum_2 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == 'op'), '2'].sum()
                    w_0_sum_2 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == '0'), '2'].sum()

                    s_pk_sum_3 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == 'pk'), '3'].sum()
                    s_op_sum_3 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == 'op'), '3'].sum()
                    s_0_sum_3 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == '0'), '3'].sum()
                    w_pk_sum_3 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == 'pk'), '3'].sum()
                    w_op_sum_3 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == 'op'), '3'].sum()
                    w_0_sum_3 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == '0'), '3'].sum()

                    s_pk_sum_4 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == 'pk'), '4'].sum()
                    s_op_sum_4 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == 'op'), '4'].sum()
                    s_0_sum_4 = abc_ddf.loc[(abc_ddf['seasons'] == 'summer') & (
                        abc_ddf['unit_type'] == '0'), '4'].sum()
                    w_pk_sum_4 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == 'pk'), '4'].sum()
                    w_op_sum_4 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == 'op'), '4'].sum()
                    w_0_sum_4 = abc_ddf.loc[(abc_ddf['seasons'] == 'winter') & (
                        abc_ddf['unit_type'] == '0'), '4'].sum()

                    TOD_x_1 = 100*((((60/resolution)*((abc_ddf['1_old_tariff'].sum(axis=0)) + ((abc_ddf['1_new_cost'].sum(axis=0))-(abc_ddf['1_old_cost'].sum(axis=0)))*(
                        shared_savings/100))/retail_tariff_df['1']))-(s_pk_sum_1 + w_pk_sum_1 + s_op_sum_1 + w_op_sum_1 + s_0_sum_1 + w_0_sum_1))/(s_pk_sum_1 + w_pk_sum_1 - s_op_sum_1 - w_op_sum_1)
                    TOD_x_2 = 100*((((60/resolution)*((abc_ddf['2_old_tariff'].sum(axis=0)) + ((abc_ddf['2_new_cost'].sum(axis=0))-(abc_ddf['2_old_cost'].sum(axis=0)))*(
                        shared_savings/100))/retail_tariff_df['2']))-(s_pk_sum_2 + w_pk_sum_2 + s_op_sum_2 + w_op_sum_2 + s_0_sum_2 + w_0_sum_2))/(s_pk_sum_2 + w_pk_sum_2 - s_op_sum_2 - w_op_sum_2)
                    TOD_x_3 = 100*((((60/resolution)*((abc_ddf['3_old_tariff'].sum(axis=0)) + ((abc_ddf['3_new_cost'].sum(axis=0))-(abc_ddf['3_old_cost'].sum(axis=0)))*(
                        shared_savings/100))/retail_tariff_df['3']))-(s_pk_sum_3 + w_pk_sum_3 + s_op_sum_3 + w_op_sum_3 + s_0_sum_3 + w_0_sum_3))/(s_pk_sum_3 + w_pk_sum_3 - s_op_sum_3 - w_op_sum_3)
                    TOD_x_4 = 100*((((60/resolution)*((abc_ddf['4_old_tariff'].sum(axis=0)) + ((abc_ddf['4_new_cost'].sum(axis=0))-(abc_ddf['4_old_cost'].sum(axis=0)))*(
                        shared_savings/100))/retail_tariff_df['4']))-(s_pk_sum_4 + w_pk_sum_4 + s_op_sum_4 + w_op_sum_4 + s_0_sum_4 + w_0_sum_4))/(s_pk_sum_4 + w_pk_sum_4 - s_op_sum_4 - w_op_sum_4)

                    TOD_x_df = pd.DataFrame(
                        [TOD_x_1, TOD_x_2, TOD_x_3, TOD_x_4])
                    TOD_x_df.to_excel(
                        f'media/outputs/{folderId}/TOD_Surcharge_Rebate.xlsx', engine='openpyxl')
                    outputLinks['TOD_Surcharge_Rebate.xlsx'] = f'media/outputs/{folderId}/TOD_Surcharge_Rebate.xlsx'
                    # end  algorithm
                    JsonData = json.dumps(outputLinks)

                    isDelFile = Files.objects.get(id=EvDataDBObjects.fileId)
                    isDelFile.delete()

                    # Calculated total time
                    totalTime = ((time.time() - int_time)/60)
                    analysisStatus = "Success"
                    errorLog = "no-error"
                    print(f"END TIME ", totalTime, userName,
                          analysisStatus, errorLog, JsonData)
                    # userAnalysis
                    userAnalysis.objects.create(
                        userName=userName, status=analysisStatus, errorLog=errorLog, time=totalTime)
                    return Response(JsonData, status=status.HTTP_201_CREATED)
                except Exception as e:
                    isDelFile = Files.objects.get(id=EvDataDBObjects.fileId)
                    isDelFile.delete()
                    # userAnalysis-error
                    analysisStatus = "Failed"
                    errorLog = e
                    jsonErrorMsg = json.dumps(str({"errMessage": e}))
                    totalTime = ((time.time() - int_time)/60)
                    userAnalysis.objects.create(
                        userName=userName, status=analysisStatus, errorLog=errorLog, time=totalTime)
                    return Response(jsonErrorMsg, status=status.HTTP_400_BAD_REQUEST)
        else:
            print(EvAnalysis.errors)
            return Response(EvAnalysis.errors, status=status.HTTP_400_BAD_REQUEST)
