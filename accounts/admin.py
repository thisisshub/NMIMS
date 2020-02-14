from django.contrib import admin

from .models import *

# Make admin classes here.
class ProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'profession', 'phone']

class AddressAdmin(admin.ModelAdmin):
    list_display = ['user', 'locality', 'city', 'state']

# Register your models here.

admin.site.register(Profile, ProfileAdmin)
admin.site.register(Address, AddressAdmin)