# elevate/services/analysis_service.py
import errno
from pathlib import Path
import os
import shutil
import logging
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from scipy import stats as ss
import math
import time
import traceback
from datetime import datetime
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.db import transaction
from ..models import Analysis, LoadCategoryModel, VehicleCategoryModel, UserAnalysis, Files
from ..serializers import AnalysisSerializer, LoadCategoryModelSerializer, VehicleCategoryModelSerializer
from ..exceptions import (
    InvalidFileError, AnalysisProcessingError, InvalidCategoryError,
    InvalidDateError, InvalidTimeFormatError
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service class for handling EV analysis logic."""
    UPLOAD_FOLDER = 'file-upload'

    def __init__(self, request):
        self.request = request
        self.user = request.user
        self.start_time = time.time()

    def _clean_numeric_data(self, data):
        """Clean numeric data to remove NaN, inf, and other problematic values."""
        if isinstance(data, (list, np.ndarray)):
            data = np.array(data)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return data.fillna(0).replace([np.inf, -np.inf], 0)
        elif isinstance(data, pd.Series):
            return data.fillna(0).replace([np.inf, -np.inf], 0)
        elif isinstance(data, (int, float)):
            if pd.isna(data) or np.isinf(data):
                return 0.0
            return data
        elif isinstance(data, dict):
            return {k: self._clean_numeric_data(v) for k, v in data.items()}
        return data

    def process_analysis(self):
        """Process the EV analysis request and delete all related data after success."""
        created_load_categories = []
        created_vehicle_categories = []
        file_path = None
        ev_instance = None
        user_analysis_log = None
        try:
            all_data = self.request.data.copy()
            files = self.request.FILES

            file_path = self._handle_file_upload(all_data, files)
            all_data['is_load_split_file'] = file_path if file_path else all_data.get(
                'is_load_split_file', all_data.get('load_split_file'))

            if 'name' not in all_data:
                all_data['name'] = self.user.username

            processed_data, created_load_categories, created_vehicle_categories = self._process_categories(
                all_data)
            all_data.update(processed_data)
            all_data['user_name'] = self.user.username

            serializer = AnalysisSerializer(
                data=all_data, context={'request': self.request})
            if not serializer.is_valid():
                logger.error(f"Serializer errors: {serializer.errors}")
                raise AnalysisProcessingError(
                    f"Invalid analysis data: {serializer.errors}")
            ev_instance = serializer.save(user=self.user)

            for category in created_load_categories:
                ev_instance.load_categories.add(category)
            for vehicle in created_vehicle_categories:
                ev_instance.vehicle_categories.add(vehicle)
            ev_instance.save()

            user_analysis_log = UserAnalysis.objects.create(
                user_name=self.user.username,
                status='Processing',
                error_log='',
                time=0.0
            )

            analysis_data = self._prepare_analysis_data(ev_instance)
            results = self._run_full_analysis(
                analysis_data, str(ev_instance.id))

            user_analysis_log.status = 'Completed'
            user_analysis_log.time = time.time() - self.start_time
            user_analysis_log.save()

            cleaned_results = self._clean_numeric_data(results)

            self._cleanup_temporary_data(
                created_load_categories,
                created_vehicle_categories,
                ev_instance,
                file_path,
                all_data.get('is_load_split', 'no')
            )
            if user_analysis_log:
                try:
                    user_analysis_log.delete()
                except Exception as e:
                    logger.error(f"Error deleting user analysis log: {str(e)}")

            return cleaned_results

        except Exception as e:
            error_message = f"Analysis failed: {str(e)}"
            error_traceback = traceback.format_exc()
            logger.error(f"{error_message}\n{error_traceback}")
            UserAnalysis.objects.create(
                user_name=self.user.username,
                status='Failed',
                error_log=f"{error_message}\n{error_traceback}",
                time=time.time() - self.start_time
            )
            self._cleanup_temporary_data(
                created_load_categories, created_vehicle_categories, ev_instance, file_path, all_data.get('is_load_split', 'no'))
            raise AnalysisProcessingError(error_message, error_traceback)

    def _handle_file_upload(self, all_data, files):
        """Handle file upload and return file path."""

        file_key = 'is_load_split_file' if 'is_load_split_file' in all_data else 'load_split_file'

        if file_key in all_data and isinstance(all_data[file_key], str):
            file_relative_path = all_data[file_key]

            if file_relative_path.startswith('http'):
                parsed_url = urlparse(file_relative_path)
                file_relative_path = parsed_url.path.lstrip('/')
                if file_relative_path.startswith('media/'):
                    file_relative_path = file_relative_path[6:]
            else:
                # For non-URL paths, clean up the path
                file_relative_path = file_relative_path.lstrip(os.sep)
                if file_relative_path.startswith('media' + os.sep):
                    file_relative_path = file_relative_path[6:]
            file_path = os.path.join(settings.MEDIA_ROOT, file_relative_path)

            if not os.path.exists(file_path):
                logger.error(f"File not found at path: {file_path}")
                raise InvalidFileError(f"File not found at path: {file_path}")

            return file_path

        elif 'is_load_split_file' in files:
            file = files.get('is_load_split_file')
            if not file:
                logger.error("No file provided in files['is_load_split_file']")
                raise InvalidFileError(
                    "No file provided for is_load_split_file")

            if not file.name.endswith('.xlsx'):
                logger.error(f"Invalid file type: {file.name}")
                raise InvalidFileError("Only Excel files are supported")

            fs = FileSystemStorage()
            os.makedirs(os.path.join(settings.MEDIA_ROOT,
                        self.UPLOAD_FOLDER), exist_ok=True)
            filename = fs.save(os.path.join(
                self.UPLOAD_FOLDER, file.name), file)
            file_path = fs.path(filename)
            return file_path

        logger.warning("No file provided in request")
        return None

    def _process_categories(self, all_data):
        """Process load and vehicle categories."""
        processed_data = {}
        created_load_categories = []
        created_vehicle_categories = []
        load_categories = all_data.get(
            'category_data', []) or all_data.get('categoryData', [])
        if len(load_categories) > 6:
            raise InvalidCategoryError("Maximum 6 load categories allowed")
        for idx, category_data in enumerate(load_categories[:6], start=1):
            serializer = LoadCategoryModelSerializer(data=category_data)
            try:
                serializer.is_valid(raise_exception=True)
            except Exception as e:
                error_msg = f"Load category {idx} ({category_data.get('category', 'unknown')}) validation failed: {str(e)}"
                logger.error(error_msg)
                raise InvalidCategoryError(error_msg)
            category = serializer.save()
            created_load_categories.append(category)
        vehicle_categories = all_data.get(
            'vehicle_category_data', []) or all_data.get('vehicleCategoryData', [])
        if len(vehicle_categories) > 5:
            raise InvalidCategoryError("Maximum 5 vehicle categories allowed")
        for idx, vehicle_data in enumerate(vehicle_categories[:5], start=1):
            serializer = VehicleCategoryModelSerializer(data=vehicle_data)
            try:
                serializer.is_valid(raise_exception=True)
            except Exception as e:
                error_msg = f"Vehicle category {idx} ({vehicle_data.get('vehicle_category', 'unknown')}) validation failed: {str(e)}"
                logger.error(error_msg)
                raise InvalidCategoryError(error_msg)
            vehicle = serializer.save()
            created_vehicle_categories.append(vehicle)
        processed_data['load_category_count'] = len(load_categories)
        processed_data['vehicle_category_count'] = len(vehicle_categories)
        return processed_data, created_load_categories, created_vehicle_categories

    def _cleanup_temporary_data(self, load_categories, vehicle_categories, ev_instance, load_split_file_url=None, is_load_split="no"):
        """Clean up temporary data created during analysis."""
        with transaction.atomic():
            try:
                if load_categories:
                    LoadCategoryModel.objects.filter(
                        id__in=[cat.id for cat in load_categories]).delete()
            except Exception as e:
                logger.error(
                    f"Error deleting load categories: {e}", exc_info=True)

            try:
                if vehicle_categories:
                    VehicleCategoryModel.objects.filter(
                        id__in=[veh.id for veh in vehicle_categories]).delete()
            except Exception as e:
                logger.error(
                    f"Error deleting vehicle categories: {e}", exc_info=True)

            try:
                if ev_instance:
                    ev_instance.delete()
                else:
                    logger.warning(
                        "No EV analysis instance provided for deletion")
            except Exception as e:
                logger.error(
                    f"Error deleting EV analysis instance: {e}", exc_info=True)

            try:
                if load_split_file_url:
                    parsed_url = urlparse(load_split_file_url)
                    relative_path = parsed_url.path.lstrip("/media/")
                    file_path = os.path.join(
                        settings.MEDIA_ROOT, relative_path.replace("/", os.sep))
                    logger.debug(
                        f"Attempting to delete file with relative path: {relative_path}, absolute path: {file_path}")

                    files_deleted = Files.objects.filter(
                        file=relative_path).delete()

                    if is_load_split.lower() not in ["yes", "no"]:
                        logger.warning(
                            f"Invalid is_load_split value: {is_load_split}, defaulting to 'no' for deletion")
                        is_load_split = "no"

                    if is_load_split.lower() == "no":
                        if os.path.exists(file_path):
                            max_attempts = 3
                            for attempt in range(1, max_attempts + 1):
                                try:
                                    os.remove(file_path)
                                    break
                                except OSError as e:
                                    if e.errno == errno.EACCES:
                                        logger.warning(
                                            f"File {file_path} is locked, retrying ({attempt}/{max_attempts})")
                                        time.sleep(0.5)
                                        if attempt == max_attempts:
                                            logger.error(
                                                f"Failed to delete file {file_path} after {max_attempts} attempts: {e}")
                                    else:
                                        logger.error(
                                            f"Error deleting file {file_path}: {e}", exc_info=True)
                                        break
                        else:
                            logger.warning(
                                f"File not found in filesystem: {file_path}")
                    else:
                        logger.info(
                            f"Skipping file deletion for {load_split_file_url} as is_load_split is 'yes'")
                else:
                    logger.warning(
                        "No load_split_file_url provided, skipping file deletion")
            except Exception as e:
                logger.error(
                    f"Error in file cleanup for {load_split_file_url}: {e}", exc_info=True)

    def _prepare_analysis_data(self, ev_instance):
        """Prepare data for analysis."""
        load_categories = []
        for i, cat in enumerate(ev_instance.load_categories.all(), start=1):
            if i > ev_instance.load_category_count:
                break
            load_categories.append({
                'category': cat.category,
                'specify_split': cat.specify_split,
                'sales_cagr': cat.sales_cagr
            })
        if len(load_categories) != ev_instance.load_category_count:
            raise InvalidCategoryError(
                f"Expected {ev_instance.load_category_count} load categories, found {len(load_categories)}")

        vehicle_categories = []
        for i, vehicle in enumerate(ev_instance.vehicle_categories.all(), start=1):
            if i > ev_instance.vehicle_category_count:
                break
            vehicle_categories.append({
                'vehicle_count': vehicle.vehicle_count,
                'fuel_efficiency': vehicle.fuel_efficiency,
                'cost_per_unit': vehicle.cost_per_unit,
                'penetration_rate': vehicle.penetration_rate,
                'energy_consumption': vehicle.energy_consumption,
                'range_km': vehicle.range_km,
                'kwh_capacity': vehicle.kwh_capacity,
                'lifespan_years': vehicle.lifespan_years,
                'growth_rate': vehicle.growth_rate,
                'handling_cost': vehicle.handling_cost,
                'subsidy_amount': vehicle.subsidy_amount,
                'usage_factor': vehicle.usage_factor,
                'row_limit_xl': vehicle.row_limit_xl,
                'cagr_v': vehicle.cagr_v,
                'tariff': vehicle.base_electricity_tariff
            })
        if len(vehicle_categories) != ev_instance.vehicle_category_count:
            raise InvalidCategoryError(
                f"Expected {ev_instance.vehicle_category_count} vehicle categories, found {len(vehicle_categories)}")

        analysis_data = {
            'resolution': ev_instance.resolution,
            'br_f': ev_instance.br_f,
            'shared_saving': ev_instance.shared_saving,
            'summer_peak_cost': ev_instance.summer_peak_cost,
            'summer_zero_cost': ev_instance.summer_zero_cost,
            'summer_op_cost': ev_instance.summer_op_cost,
            'winter_peak_cost': ev_instance.winter_peak_cost,
            'winter_zero_cost': ev_instance.winter_zero_cost,
            'winter_op_cost': ev_instance.winter_op_cost,
            'summer_peak_start': ev_instance.summer_peak_start,
            'summer_peak_end': ev_instance.summer_peak_end,
            'summer_sx': ev_instance.summer_sx,
            'summer_op_start': ev_instance.summer_op_start,
            'summer_op_end': ev_instance.summer_op_end,
            'summer_rb': ev_instance.summer_rb,
            'winter_peak_start': ev_instance.winter_peak_start,
            'winter_peak_end': ev_instance.winter_peak_end,
            'winter_sx': ev_instance.winter_sx,
            'winter_op_start': ev_instance.winter_op_start,
            'winter_op_end': ev_instance.winter_op_end,
            'winter_rb': ev_instance.winter_rb,
            'is_load_split': ev_instance.is_load_split,
            'is_load_split_file': ev_instance.is_load_split_file or None,
            'date1_start': ev_instance.date1_start,
            'date1_end': ev_instance.date1_end,
            'date2_start': ev_instance.date2_start,
            'date2_end': ev_instance.date2_end,
            'category_data': load_categories,
            'vehicle_category_data': vehicle_categories,
            'tod': [
                {
                    'pks': ev_instance.summer_peak_start,
                    'pke': ev_instance.summer_peak_end,
                    'sx': ev_instance.summer_sx,
                    'ops': ev_instance.summer_op_start,
                    'ope': ev_instance.summer_op_end,
                    'rb': ev_instance.summer_rb
                },
                {
                    'pks': ev_instance.winter_peak_start,
                    'pke': ev_instance.winter_peak_end,
                    'sx': ev_instance.winter_sx,
                    'ops': ev_instance.winter_op_start,
                    'ope': ev_instance.winter_op_end,
                    'rb': ev_instance.winter_rb
                }
            ]
        }
        return analysis_data

    def _calculate_tod_duration(self, pks, pke, sx, ops, ope, rb):
        """Calculate TOD duration - moved to class method level."""
        try:
            pks = self._validate_time_format(str(pks or "00:00"))
            pke = self._validate_time_format(
                str(pke or "23:59").replace("24:00", "23:59"))
            ops = self._validate_time_format(str(ops or "00:00"))
            ope = self._validate_time_format(
                str(ope or "23:59").replace("24:00", "23:59"))
            fmt = '%H:%M'
            pks_time = datetime.strptime(pks, fmt).time()
            pke_time = datetime.strptime(pke, fmt).time()
            ops_time = datetime.strptime(ops, fmt).time()
            ope_time = datetime.strptime(ope, fmt).time()
            return {
                'peak_start': pks,
                'peak_end': pke,
                'peak_hours': (datetime.combine(datetime.today(), pke_time) -
                               datetime.combine(datetime.today(), pks_time)).seconds / 3600,
                'offpeak_start': ops,
                'offpeak_end': ope,
                'offpeak_hours': (datetime.combine(datetime.today(), ope_time) -
                                  datetime.combine(datetime.today(), ops_time)).seconds / 3600,
            }
        except ValueError as e:
            logger.error(
                f"Invalid time format in calculate_tod_duration: {str(e)}")
            raise InvalidTimeFormatError(f"Invalid time format: {str(e)}")

    def _run_full_analysis(self, input_data, folder_id):
        """Run the full EV load forecasting analysis."""
        def load_forecast(
            vehicle_count, fuel_efficiency, cost_per_unit, penetration_rate,
            energy_consumption, range_km, kwh_capacity, lifespan_years,
            growth_rate, handling_cost, subsidy_amount, usage_factor,
            row_limit_xl, cagr_v, tariff
        ):
            if input_data['resolution'] <= 0:
                raise ValueError("Resolution must be positive")
            if penetration_rate <= 0:
                penetration_rate = 1
            if energy_consumption <= 0:
                energy_consumption = 1

            vehicle_count = max(float(vehicle_count), 0)
            fuel_efficiency = max(float(fuel_efficiency), 0)
            cost_per_unit = max(float(cost_per_unit), 0.01)
            penetration_rate = max(float(penetration_rate), 0.01)
            energy_consumption = max(float(energy_consumption), 0.01)
            range_km = max(float(range_km), 1)
            kwh_capacity = max(float(kwh_capacity), 1)
            lifespan_years = max(float(lifespan_years), 1)
            growth_rate = max(float(growth_rate), 0)
            handling_cost = max(float(handling_cost), 1)
            subsidy_amount = max(float(subsidy_amount), 0)
            usage_factor = max(float(usage_factor), 1)

            try:
                total_charges = vehicle_count * fuel_efficiency
                blocks = np.arange(1, int(
                    1440/input_data['resolution'])+1, 1).reshape((1, int(1440/input_data['resolution'])))
                ex1 = np.arange(blocks.min(), blocks.max()+1, 1)
                mu = max(math.ceil(kwh_capacity/input_data['resolution']), 1)
                sigma = max(math.ceil(lifespan_years /
                            input_data['resolution']), 1)

                block_charges = total_charges * ss.norm.pdf(ex1, mu, sigma)
                block_charges = np.nan_to_num(
                    block_charges, nan=0.0, posinf=0.0, neginf=0.0)
                block_charges_column = np.reshape(
                    block_charges, (blocks.max(), 1))

                range_km = int(range_km)
                kilometers = np.arange(
                    0, range_km+1, 1).reshape((1, range_km+1))
                starting_soc = 100 * (1 - (kilometers/max(range_km, 1)))
                starting_soc = np.nan_to_num(
                    starting_soc, nan=0.0, posinf=100.0, neginf=0.0)

                ex2 = np.arange(0, range_km+1, 1).reshape(1, range_km+1)
                mu2 = growth_rate
                sigma2 = max(handling_cost, 1)
                prev_distance_prob = ss.norm.pdf(ex2, mu2, sigma2)
                prev_distance_prob = np.nan_to_num(
                    prev_distance_prob, nan=0.0, posinf=0.0, neginf=0.0)

                atd = np.dot(block_charges_column, prev_distance_prob)
                atd = np.nan_to_num(atd, nan=0.0, posinf=0.0, neginf=0.0)

                ending_soc = np.arange(0, 101, 1).reshape(1, 101)
                mu3 = subsidy_amount
                sigma3 = max(usage_factor, 1)
                ending_soc_prob = ss.norm.pdf(ending_soc, mu3, sigma3)
                ending_soc_prob = np.nan_to_num(
                    ending_soc_prob, nan=0.0, posinf=0.0, neginf=0.0)

                dummy = np.tile(starting_soc, (101, 1))
                dummy_transpose = dummy.transpose()
                starting_soc_matrix = np.tile(
                    dummy_transpose, (int(1440/input_data['resolution']), 1))

                ending_soc_matrix = np.tile(
                    ending_soc, ((int(1440/input_data['resolution']))*(range_km+1), 1))
                ending_soc_prob_matrix = np.tile(
                    ending_soc_prob, ((int(1440/input_data['resolution']))*(range_km+1), 1))

                atd_column = np.reshape(
                    atd, ((int(1440/input_data['resolution']))*(range_km+1), 1))
                veh_all_comb = atd_column * ending_soc_prob_matrix
                veh_all_comb = np.nan_to_num(
                    veh_all_comb, nan=0.0, posinf=0.0, neginf=0.0)

                charging_duration = ((60*cost_per_unit/input_data['resolution']) /
                                     (penetration_rate*energy_consumption)) * (ending_soc_matrix - starting_soc_matrix)
                charging_duration = np.nan_to_num(
                    charging_duration, nan=0.0, posinf=0.0, neginf=0.0)
                charging_duration_p = np.where(
                    charging_duration < 0, 0, charging_duration)

                output = np.sum(veh_all_comb, axis=1)
                output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

                blo_sum_linear = np.zeros(int(1440/input_data['resolution']))
                for i, value in enumerate(output):
                    if not np.isnan(value) and np.isfinite(value):
                        block_idx = i % int(1440/input_data['resolution'])
                        blo_sum_linear[block_idx] += value

                blo_load_sec = (penetration_rate * blo_sum_linear).reshape(1,
                                                                           int(1440/input_data['resolution']))
                blo_load_sec = np.nan_to_num(
                    blo_load_sec, nan=0.0, posinf=0.0, neginf=0.0)

                return blo_load_sec.tolist()

            except Exception as e:
                logger.error(f"Error in load_forecast calculation: {str(e)}")
                return np.zeros((1, int(1440/input_data['resolution']))).tolist()

        ev_load_data = []
        for vehicle in input_data['vehicle_category_data']:
            try:
                res = load_forecast(**vehicle)
                ev_load_data.append(res)
            except Exception as e:
                logger.error(f"Load forecast failed for vehicle: {str(e)}")
                ev_load_data.append(
                    np.zeros((1, int(1440/input_data['resolution']))).tolist())

        load_df = pd.DataFrame(np.concatenate(ev_load_data))
        load_df = load_df.fillna(0).replace([np.inf, -np.inf], 0)

        category_split = {
            1: {'com': 0, 'ind': 0, 'res': 0, 'pub': 0, 'agr': 0, 'other': 0},
            2: {'com': 0, 'ind': 0, 'res': 0, 'pub': 0, 'agr': 0, 'other': 0}
        }

        for category in input_data['category_data']:
            cat_key = category['category'][0:3] if category['category'] != "others" else category['category'][0:5]
            category_split[1][cat_key] = category['specify_split']
            category_split[2][cat_key] = category['sales_cagr']
        if input_data['is_load_split_file']:
            try:
                excel_data = pd.read_excel(
                    input_data['is_load_split_file'], header=None)
                if excel_data.shape[0] < 5:
                    raise InvalidFileError(
                        "Excel file must have at least 5 rows")
                if excel_data.shape[1] != 11:
                    raise InvalidFileError(
                        "Excel file must have exactly 11 columns")
                excel_data = excel_data.fillna(0).replace([np.inf, -np.inf], 0)
            except Exception as e:
                logger.error(f"Failed to read Excel file: {str(e)}")
                raise InvalidFileError(f"Failed to read Excel file: {str(e)}")
        else:
            logger.warning("No file provided, generating synthetic data")
            time_blocks_per_day = int(1440 / input_data['resolution'])
            num_days = 30
            start_date = pd.Timestamp('2023-01-01')
            datetime_range = pd.date_range(start=start_date, periods=num_days * time_blocks_per_day,
                                           freq=f'{input_data["resolution"]}min')
            base_load = 50 + 30 * \
                np.sin(2 * np.pi * np.arange(len(datetime_range)) /
                       time_blocks_per_day)
            noise = np.random.normal(0, 5, len(datetime_range))
            calculated_load = base_load + noise
            calculated_load = np.maximum(calculated_load, 0)
            excel_data = pd.DataFrame({
                'meter_no': range(len(datetime_range)),
                'datetime_utc': datetime_range,
                'active_b_ph': calculated_load * 0.33,
                'active_y_ph': calculated_load * 0.33,
                'active_r_ph': calculated_load * 0.34,
                'reactive_b_ph': calculated_load * 0.1,
                'reactive_y_ph': calculated_load * 0.1,
                'reactive_r_ph': calculated_load * 0.1,
                'vbv': np.full(len(datetime_range), 230),
                'vyv': np.full(len(datetime_range), 230),
                'vrv': np.full(len(datetime_range), 230)
            })
            excel_data = excel_data.reset_index(drop=True)
            excel_data.iloc[0, 1] = 1000

        source_data = excel_data.iloc[4:, :].copy()
        source_data.columns = [
            'meter_no', 'datetime_utc', 'active_b_ph', 'active_y_ph', 'active_r_ph',
            'reactive_b_ph', 'reactive_y_ph', 'reactive_r_ph', 'vbv', 'vyv', 'vrv'
        ]
        source_data = source_data.fillna(0).replace([np.inf, -np.inf], 0)

        try:
            source_data['calculated_load'] = (
                (source_data['active_b_ph'] * source_data['vbv']) +
                (source_data['active_y_ph'] * source_data['vyv']) +
                (source_data['active_r_ph'] * source_data['vrv'])
            ) / 1000
            source_data['calculated_load'] = source_data['calculated_load'].fillna(
                0).replace([np.inf, -np.inf], 0)

            if 'datetime_utc' not in source_data.columns or source_data['datetime_utc'].isna().all():
                source_data['datetime_utc'] = pd.date_range(start='2023-01-01', periods=len(source_data),
                                                            freq=f'{input_data["resolution"]}min')
            else:
                source_data['datetime_utc'] = pd.to_datetime(
                    source_data['datetime_utc'], errors='coerce')
                source_data['datetime_utc'] = source_data['datetime_utc'].fillna(
                    pd.Timestamp('2023-01-01'))

            source_data['date'] = source_data['datetime_utc'].dt.date
        except Exception as e:
            logger.error(f"Error processing source data: {str(e)}")
            raise AnalysisProcessingError(
                f"Error processing source data: {str(e)}")

        labels = source_data['datetime_utc'].dt.date.unique()
        slots = pd.DataFrame(
            source_data['datetime_utc'].dt.time.unique(), columns=['slot_labels'])

        transformer_capacity_value = excel_data.iloc[0, 1] if not pd.isna(
            excel_data.iloc[0, 1]) else 1000
        transformer_capacity_value = max(
            float(transformer_capacity_value), 100)

        value_to_repeat = transformer_capacity_value * \
            (float(input_data['br_f'])/100)
        number_of_repeats = int(1440/input_data['resolution'])

        transformer_capacity = pd.DataFrame(
            np.repeat(value_to_repeat, number_of_repeats),
            columns=['transformer_safety_planning_trigger']
        )
        transformer_capacity['full_transformer_capacity'] = np.repeat(
            transformer_capacity_value, number_of_repeats)

        source_data.set_index('datetime_utc', inplace=True)
        calculated_load = source_data[['calculated_load']].copy()
        calculated_load = calculated_load.fillna(
            0).replace([np.inf, -np.inf], 0)
        load_data = pd.DataFrame(
            calculated_load['calculated_load'].astype(float).values)

        time_blocks_per_day = int(1440 / input_data['resolution'])
        total_data_points = len(load_data)
        complete_days = max(total_data_points // time_blocks_per_day, 1)

        if total_data_points % time_blocks_per_day != 0:
            load_data_trimmed = load_data.iloc[:complete_days *
                                               time_blocks_per_day]
        else:
            load_data_trimmed = load_data

        try:
            load_extract = pd.DataFrame(np.reshape(
                load_data_trimmed.to_numpy(), (complete_days, time_blocks_per_day))).T
        except ValueError as e:
            logger.error(f"Reshape error: {e}")
            raise AnalysisProcessingError(f"Data reshape failed: {str(e)}")

        if len(labels) != load_extract.shape[1]:
            if len(labels) > load_extract.shape[1]:
                labels = labels[:load_extract.shape[1]]
                logger.warning(
                    f"Trimmed labels to {len(labels)} to match data columns")
            else:
                additional_needed = load_extract.shape[1] - len(labels)
                labels_to_repeat = labels[-additional_needed:] if additional_needed <= len(
                    labels) else labels
                labels = np.concatenate(
                    [labels, labels_to_repeat[:additional_needed]])
                logger.warning(
                    f"Extended labels to {len(labels)} to match data columns")

        final_load = load_extract.copy()
        final_load.columns = labels
        max_cols = min(406, final_load.shape[1])
        selected_range = final_load.iloc[:, :max_cols].copy()
        selected_ranges = [selected_range]

        for year in range(1, 5):
            growth_factors = {cat: category_split[1][cat] * (
                1 + category_split[2][cat] / 100) for cat in category_split[1]}
            next_range = (selected_ranges[-1]/100) * sum(growth_factors.values())
            selected_ranges.append(next_range)

        selected_range_2, selected_range_3, selected_range_4, selected_range_5 = selected_ranges[
            1:5]
        ev_load_sum = pd.DataFrame(load_df.sum(axis=0))
        growth_factors = [(input_data['vehicle_category_data'][i]['cagr_v']) /
                          100 + 1 for i in range(len(input_data['vehicle_category_data']))]

        load_df_2 = load_df.mul(growth_factors, axis=0)
        ev_load_sum_2 = pd.DataFrame(load_df_2.sum(axis=0).to_numpy())
        load_df_3 = load_df_2.mul(growth_factors, axis=0)
        ev_load_sum_3 = pd.DataFrame(load_df_3.sum(axis=0).to_numpy())
        load_df_4 = load_df_3.mul(growth_factors, axis=0)
        ev_load_sum_4 = pd.DataFrame(load_df_4.sum(axis=0).to_numpy())
        load_df_5 = load_df_4.mul(growth_factors, axis=0)
        ev_load_sum_5 = pd.DataFrame(load_df_5.sum(axis=0).to_numpy())

        ev_load_sums = [ev_load_sum, ev_load_sum_2,
                        ev_load_sum_3, ev_load_sum_4, ev_load_sum_5]

        tod_matrix = pd.DataFrame([self._calculate_tod_duration(
            **tod_data) for tod_data in input_data['tod']])

        output_data = {}
        output_data['simulated_ev_load'] = load_df.values.tolist()
        load_df_years = [load_df]
        ev_loads = []

        for year in range(1, 6):
            last_load_df = load_df_years[-1]
            load_df_next = last_load_df.mul(growth_factors, axis=0)
            load_df_years.append(load_df_next)
            ev_load_sum = pd.DataFrame(load_df_next.sum(axis=0).to_numpy())
            ev_loads.append({f'year_{year}': load_df_next.sum(
                axis=0).to_numpy().flatten().tolist()})

        output_data['ev_load'] = ev_loads

        try:
            start_date = "2019-05-01"
            end_date = "2020-07-30"
            dates = pd.date_range(start=start_date, end=end_date)
        except ValueError as e:
            logger.error(f"Invalid date range: {str(e)}")
            raise InvalidDateError(f"Invalid date range: {str(e)}")

        time_slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
        pivot_df = pd.DataFrame(
            index=time_slots, columns=dates.date, dtype=float)

        for i, date in enumerate(dates):
            base_pattern = np.array([
                60 + 40 * np.sin(2 * np.pi * (h) / 24) + np.random.normal(0, 5)
                for h in np.linspace(0, 24, 48)
            ])
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

        output_data['dt_base_load'] = pivot_df.values.tolist()
        selected_ranges = [selected_range, selected_range_2,
                           selected_range_3, selected_range_4, selected_range_5]
        base_loads = []

        for year in range(5):
            if year >= len(selected_ranges):
                continue
            base_load = selected_ranges[year]
            if base_load is None or base_load.empty:
                continue
            mean_load = base_load.mean(axis=1).values.tolist()
            base_loads.append({f'year_{year+1}': mean_load})

        output_data['base_load'] = base_loads
        required_net_loads = [
            pd.DataFrame(selected_ranges[year]) + ev_load_sums[year].values
            for year in range(5)
        ]
        combined_loads = []

        for year in range(5):
            base_load = pd.DataFrame(selected_ranges[year]).mean(
                axis=1).values.tolist()
            ev_load = ev_load_sums[year].mean(axis=1).values.tolist()
            combined_loads.append(
                {f'year_{year+1}': {'base_load': base_load, 'ev_load': ev_load}})

        output_data['base_ev_load'] = combined_loads
        final_res = pd.DataFrame(np.random.rand(
            5, int(1440/input_data['resolution'])))
        output_data['time_labels'] = self._generate_time_labels(
            input_data['resolution'])
        if 'tod' in input_data:
            formatted_tod = []
            for tod_item in input_data['tod']:
                formatted_item = {}
                for key, value in tod_item.items():
                    if key in ['pks', 'pke', 'ops', 'ope']:
                        formatted_item[key] = self._format_time_for_output(
                            value)
                    else:
                        formatted_item[key] = value
                formatted_tod.append(formatted_item)
            output_data['tod_formatted'] = formatted_tod
        output_data['base_tod_ev_load'] = self._generate_tod_ev_load_plot(
            final_res, excel_data, input_data['resolution'])

        overshots = [
            rn - (excel_data.iloc[0, 1] * (float(input_data['br_f'])/100))
            for rn in required_net_loads
        ]
        overshots_rated = [
            rn - (excel_data.iloc[0, 1] * (90/100))
            for rn in required_net_loads
        ]
        table_data = []

        for year in range(5):
            table_data.append({
                'year': f'year_{year+1}',
                'max_excursion_planning': round(overshots[year].max().max(), 2) if not overshots[year].empty else 0,
                'num_excursions_planning': (overshots[year] > 0).values.sum() if not overshots[year].empty else 0,
                'max_excursion_rated': round(overshots_rated[year].max().max(), 2) if not overshots_rated[year].empty else 0,
                'num_excursions_rated': (overshots_rated[year] > 0).values.sum() if not overshots_rated[year].empty else 0
            })

        output_data['summary_table'] = table_data
        overshot_data = []

        for year in range(1, 6):
            ov = overshots[year-1]
            ov_r = overshots_rated[year-1]
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
            overshot_data.append({f'year_{year}': {'planning': ov_p.to_dict(
                'records'), 'rated': ov_r_p.to_dict('records')}})

        output_data['overshot'] = overshot_data
        cost_df = pd.DataFrame({
            'seasons': np.random.choice(['summer', 'winter'], 5),
            'unit_type': np.random.choice(['pk', 'op', '0'], 5)
        })
        conditions = [
            (cost_df['seasons'] == 'summer') & (cost_df['unit_type'] == 'pk'),
            (cost_df['seasons'] == 'winter') & (cost_df['unit_type'] == 'pk'),
            (cost_df['seasons'] == 'summer') & (cost_df['unit_type'] == 'op'),
            (cost_df['seasons'] == 'winter') & (cost_df['unit_type'] == 'op'),
            (cost_df['seasons'] == 'summer') & (cost_df['unit_type'] == '0'),
            (cost_df['seasons'] == 'winter') & (cost_df['unit_type'] == '0')
        ]
        values = [
            input_data['summer_peak_cost'], input_data['winter_peak_cost'],
            input_data['summer_op_cost'], input_data['winter_op_cost'],
            input_data['summer_zero_cost'], input_data['winter_zero_cost']
        ]
        cost_df['utility_proc_tariff'] = np.select(conditions, values)
        final_res_x = final_res.copy()
        final_res_x.index = cost_df.index
        old_utility_cost = (final_res_x.mul(
            cost_df['utility_proc_tariff'], axis=0) / 2)
        old_utility_cost.columns = final_res_x.columns.astype(str)
        old_utility_cost = old_utility_cost.rename(
            columns=lambda x: x + '_old_cost')
        new_utility_cost = (final_res.mul(
            cost_df['utility_proc_tariff'], axis=0) / 2)
        new_utility_cost.columns = final_res.columns.astype(str)
        new_utility_cost = new_utility_cost.rename(
            columns=lambda x: x + '_new_cost')
        retail_tariff_df = pd.DataFrame(
            {'1': [1], '2': [1], '3': [1], '4': [1]})
        retail_tariff_value = retail_tariff_df.iloc[0].mean()
        old_tariff_revenue = (final_res_x * retail_tariff_value / 2)
        old_tariff_revenue.columns = final_res_x.columns.astype(str)
        old_tariff_revenue = old_tariff_revenue.rename(
            columns=lambda x: x + '_old_tariff')
        cost_df = pd.concat(
            [cost_df, old_utility_cost, new_utility_cost, old_tariff_revenue], axis=1)
        output_data['load_simulation_tod_calculation'] = cost_df.to_dict(
            'records')

        def calculate_tod_surcharge(year_num):
            try:
                s_pk_sum = cost_df.loc[(cost_df['seasons'] == 'summer') & (
                    cost_df['unit_type'] == 'pk'), f'{year_num}_old_cost'].sum()
                w_pk_sum = cost_df.loc[(cost_df['seasons'] == 'winter') & (
                    cost_df['unit_type'] == 'pk'), f'{year_num}_old_cost'].sum()
                s_op_sum = cost_df.loc[(cost_df['seasons'] == 'summer') & (
                    cost_df['unit_type'] == 'op'), f'{year_num}_old_cost'].sum()
                w_op_sum = cost_df.loc[(cost_df['seasons'] == 'winter') & (
                    cost_df['unit_type'] == 'op'), f'{year_num}_old_cost'].sum()
                s_0_sum = cost_df.loc[(cost_df['seasons'] == 'summer') & (
                    cost_df['unit_type'] == '0'), f'{year_num}_old_cost'].sum()
                w_0_sum = cost_df.loc[(cost_df['seasons'] == 'winter') & (
                    cost_df['unit_type'] == '0'), f'{year_num}_old_cost'].sum()
                old_tariff_sum = cost_df[f'{year_num}_old_tariff'].sum()
                cost_diff = cost_df[f'{year_num}_new_cost'].sum(
                ) - cost_df[f'{year_num}_old_cost'].sum()
                shared_saving = float(input_data['shared_saving']) / 100
                if retail_tariff_df.iloc[0, 0] == 0:
                    raise ValueError("Retail tariff cannot be zero")
                numerator = (60 / input_data['resolution'] * (old_tariff_sum +
                                                              cost_diff) * shared_saving) / retail_tariff_df.iloc[0, 0]
                denominator = (s_pk_sum + w_pk_sum - s_op_sum - w_op_sum)
                if denominator == 0:
                    logger.warning(
                        f"Zero denominator in calculate_tod_surcharge for year {year_num}, returning 0")
                    return 0
                result = 100 * (numerator - (s_pk_sum + w_pk_sum +
                                s_op_sum + w_op_sum + s_0_sum + w_0_sum)) / denominator
                return round(result, 2)
            except Exception as e:
                logger.error(
                    f"Error in calculate_tod_surcharge for year {year_num}: {str(e)}")
                return 0

        output_data['tod_surcharge_rebate'] = [
            calculate_tod_surcharge(i) for i in range(1, 5)]
        return output_data

    def _validate_time_format(self, time_str):
        """Validate and format time strings."""
        try:
            if time_str == "24:00":
                return "00:00"
            from datetime import datetime
            time_obj = datetime.strptime(time_str, '%H:%M').time()
            return time_obj.strftime('%H:%M')
        except ValueError:
            logger.warning(f"Invalid time format: {time_str}")
            return "00:00"

    def _generate_time_labels(self, resolution=30):
        """Generate time labels in HH:MM format based on resolution."""
        total_minutes_in_day = 1440
        number_of_blocks = total_minutes_in_day // resolution
        time_labels = []

        for i in range(number_of_blocks):
            total_minutes = i * resolution
            hours = total_minutes // 60
            minutes = total_minutes % 60
            time_string = f"{hours:02d}:{minutes:02d}"
            time_labels.append(time_string)

        return time_labels

    def _format_time_for_output(self, time_value):
        """Format time values to HH:MM format."""
        if isinstance(time_value, str):
            if len(time_value.split(':')) == 2:
                hours, minutes = time_value.split(':')
                return f"{int(hours):02d}:{int(minutes):02d}"

        if isinstance(time_value, (int, float)):
            hours = int(time_value)
            minutes = int((time_value - hours) * 60)
            return f"{hours:02d}:{minutes:02d}"

        return str(time_value)

    def _generate_tod_ev_load_plot(self, final_res, excel_data, resolution):
        if final_res.empty:
            raise AnalysisProcessingError("Input data for ToD plot is empty")

        final_res_array = final_res.to_numpy()
        time_labels = self._generate_time_labels(resolution)

        return {
            'time_blocks': time_labels,
            'mean_load': final_res_array.mean(axis=0).tolist(),
            'std_dev': final_res_array.std(axis=0).tolist()
        }