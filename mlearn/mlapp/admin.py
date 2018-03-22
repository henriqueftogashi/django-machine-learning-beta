from django.contrib import admin
from mlapp.models import DownloadedFile, CurrentFile, Prepross

# Register your models here.
admin.site.register(DownloadedFile)
admin.site.register(CurrentFile)
admin.site.register(Prepross)
