from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

PERCENTAGE_VALIDATOR = [MinValueValidator(0), MaxValueValidator(100)]


class LoadCategoryModel(models.Model):
    CATEGORY_CHOICES = (("commercial", 'commercial'), ("agricultural", 'agricultural'), ("industrial",
                        'industrial'), ("residential", 'residential'), ("public", 'public'), ("others", 'others'),)
    category = models.CharField(
        max_length=20, choices=CATEGORY_CHOICES, default="")
    categoryFile = models.CharField(
        default="", blank=True, null=True, max_length=400)
    salesCAGR = models.IntegerField(blank=True, null=True, default=0)
    specifySplit = models.FloatField(null=True, blank=True, default=0)

    # def __str__(self):
    #     return self.id


class vehicleCategoryModel(models.Model):
    VEHICLE_CATEGORY_CHOICES = (("car", 'car'), ("bus", 'bus'), (
        "2-wheeler", '2-wheeler'), ("3-wheeler", '3-wheeler'), ("others", 'others'),)
    vehicleCategory = models.CharField(
        max_length=20, choices=VEHICLE_CATEGORY_CHOICES, default="")
    n = models.IntegerField(default=0, blank=True, null=True)
    f = models.IntegerField(default=0, blank=True, null=True)
    c = models.IntegerField(default=0, blank=True, null=True)
    p = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    e = models.FloatField(default=0.0, blank=True, null=True)
    r = models.IntegerField(default=0, blank=True, null=True)
    k = models.IntegerField(default=0, blank=True, null=True)
    l = models.IntegerField(default=0, blank=True, null=True)
    g = models.IntegerField(default=0, blank=True, null=True)
    h = models.IntegerField(default=0, blank=True, null=True)
    s = models.IntegerField(default=0, blank=True, null=True)
    u = models.IntegerField(default=0, blank=True, null=True)
    rowlimit_xl = models.IntegerField(default=2000000, blank=False, null=False)
    CAGR_V = models.IntegerField(default=0, blank=True, null=True)
    baseElectricityTariff = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)

    # n=int(input("Enter the number of vehicles of a single category")) #1000                 #Number of vehicles of a single category
    # f=int(input("Enter the frequency of charging each vehicle per day")) #1
    # c=int(input("Enter the battery capacity of the chosen category of vehicle in kwh")) #38
    # p=int(input("Enter the charging power required for the chosen category of vehicle in kw")) #10
    # e=int(input("Enter the charging efficiency % \nFor 20 percent, enter 20 without the percentage sign")) #95
    # r=int(input("Enter the chosen vehicle range in kilometers")) #300
    # k=int(input("Enter the average/ mean of charging start time in minutes. Consider a 24 hour clock. For example 6 Pm would be 1080 minutes - 18 hours x 60 minutes"))#1200
    # l=int(input("Enter the standard deviation of k in minutes")) #120
    # g=int(input("Enter the average daily trip length in kilometers")) #10
    # h=int(input("Enter the standard deviation of g in kilometers")) #3
    #s=int(input("Enter the average/mean of possible Ending State Of Charges of the vehicle batteries.For 20 percent, enter 20 without any percentage sign"))
    # u=int(input("Enter the standard deviation of s. For 20 percent, enter 20 without any percentage sign"))#2
    # rowlimit_xl=2000000     #0.25 lakh rows per excel sheet are permitted - actual limit is 1,048,576
    # CAGR_V=int(input("Enter the CAGR of sales for chosen vehicle category. For 20 percent, enter 20 without any percentage sign")) #5


class evAnalysis(models.Model):
    loadCategory = models.IntegerField(
        default=0, validators=[MinValueValidator(1), MaxValueValidator(6)])

    LOAD_SPLIT_CHOICES = (("yes", 'yes'), ("no", 'no'),)
    isLoadSplit = models.CharField(
        max_length=20, choices=LOAD_SPLIT_CHOICES, default="")

    isLoadSplitFile = models.CharField(
        default="", blank=True, null=True, max_length=400)

    loadCategory1 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, max_length=30, related_name='load_Cat_1')
    loadCategory2 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, max_length=30, related_name='load_Cat_2')
    loadCategory3 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.CASCADE, null=True, blank=True, max_length=30, related_name='load_Cat_3')
    loadCategory4 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, max_length=30, related_name='load_Cat_4')
    loadCategory5 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, max_length=30, related_name='load_Cat_5')
    loadCategory6 = models.ForeignKey(
        LoadCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, max_length=30, related_name='load_Cat_6')

    numOfvehicleCategory = models.IntegerField(blank=True, null=True, default=0, validators=[
        MinValueValidator(1), MaxValueValidator(5)])

    vehicleCategoryData1 = models.ForeignKey(
        vehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_Cat_1')
    vehicleCategoryData2 = models.ForeignKey(
        vehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_Cat_2')
    vehicleCategoryData3 = models.ForeignKey(
        vehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_Cat_3')
    vehicleCategoryData4 = models.ForeignKey(
        vehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_Cat_4')
    vehicleCategoryData5 = models.ForeignKey(
        vehicleCategoryModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='vehicle_Cat_5')

    resolution = models.IntegerField(
        default=0, validators=[MinValueValidator(1), MaxValueValidator(100)], blank=True, null=True,)
    BR_F = models.IntegerField(
        default=0, blank=True, null=True,)
    sharedSavaing = models.IntegerField(
        default=0,  blank=True, null=True,)

    sum_pk_cost = models.IntegerField(
        default=0,  blank=True, null=True,)
    sum_zero_cost = models.IntegerField(
        default=0,  blank=True, null=True,)
    sum_op_cost = models.IntegerField(
        default=0,  blank=True, null=True,)
    win_pk_cost = models.IntegerField(
        default=0,  blank=True, null=True,)
    win_zero_cost = models.FloatField(null=True, blank=True, default=0)
    win_op_cost = models.IntegerField(
        default=0,  blank=True, null=True,)

    date1_start = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date1_end = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date2_start = models.CharField(
        default="", blank=False, null=False, max_length=10)
    date2_end = models.CharField(
        default="", blank=False, null=False, max_length=10)

    # s = summer
    # w = winter

    s_pks = models.CharField(
        default="", blank=False, null=False, max_length=10)
    s_pke = models.CharField(
        default="", blank=False, null=False, max_length=10)
    s_sx = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    s_ops = models.CharField(
        default="", blank=False, null=False, max_length=10)
    s_ope = models.CharField(
        default="", blank=False, null=False, max_length=10)
    s_rb = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)

    w_pks = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_pke = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_sx = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)
    w_ops = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_ope = models.CharField(
        default="00:00", blank=False, null=False, max_length=10)
    w_rb = models.FloatField(
        default=0, validators=PERCENTAGE_VALIDATOR, blank=True)

    fileId = models.IntegerField(
        default=0,  blank=False, null=False)
    user_name = models.CharField(
        default="", blank=False, null=False, max_length=50)

    class Meta:
        verbose_name_plural = "Ev Analysis's"


class Files(models.Model):
    file = models.FileField(upload_to="FileUpload/", blank=False, null=False)

    def delete(self, *args, **kwargs):
        # You have to prepare what you need before delete the model
        storage, path = self.file.storage, self.file.path
        # Delete the model before the file
        super(Files, self).delete(*args, **kwargs)
        # Delete the file after the model
        storage.delete(path)


class userAnalysis(models.Model):
    userName = models.CharField(
        default="", blank=False, null=False, max_length=50)
    status = models.CharField(default="", blank=True, null=True, max_length=20)
    errorLog = models.TextField()
    time = models.FloatField(default=0.0, blank=True, null=True)

    class Meta:
        verbose_name_plural = "User Analysis's Log"
