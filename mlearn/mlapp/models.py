from django.db import models
from static.py.funcs import OverwriteStorage
import json

# Create your models here.
class DownloadedFile(models.Model):
    docfile = models.FileField(storage=OverwriteStorage(), upload_to='downloaded/')

class CurrentFile(models.Model):
    filename = models.CharField(max_length=300)

class Prepross(models.Model):
    filename = models.CharField(max_length=300)
    coltype = models.CharField(max_length=300)
    assvar = models.CharField(max_length=300)
    missingvalues = models.CharField(max_length=300)
    trainingset_size = models.IntegerField()
    featscaling = models.CharField(max_length=300)
