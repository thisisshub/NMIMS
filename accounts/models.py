from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Profile(models.Model):
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profession = models.CharField(max_length=50, blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    phone = models.CharField(max_length=12, blank=True, null=True)
    facebook = models.URLField(max_length=200, blank=True, null=True)
    instagram = models.URLField(max_length=200, blank=True, null=True)
    twitter = models.URLField(max_length=200, blank=True, null=True)
    linkedin = models.URLField(max_length=200, blank=True, null=True)
    github = models.URLField(max_length=200, blank=True, null=True)
    profilepic = models.ImageField(default='Pictures/profile_pics/default.png', upload_to='Pictures/profile_pics', blank='true')

    class Meta:
        verbose_name = "Profile"
        verbose_name_plural = "Profiles"

    def __str__(self):
        return str(self.user)

class Address(models.Model):

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    house_number = models.CharField(max_length=10, blank=True, null=True)
    locality = models.CharField(max_length=50, blank=True, null=True)
    landmark = models.CharField(max_length=50, blank=True, null=True)
    city = models.CharField(max_length=50, blank=True, null=True)
    state = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        verbose_name = "Address"
        verbose_name_plural = "Addresses"

    def __str__(self):
        return str(self.user)