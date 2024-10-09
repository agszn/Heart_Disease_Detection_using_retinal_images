from .models import *
from django.contrib import admin


@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display = ["id", "name", "email", "message"]

