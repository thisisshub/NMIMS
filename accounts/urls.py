from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView

from .views import *

app_name = 'profiles'

urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('registration/', UserRegistrationView.as_view(), name="registration"),
    path('registration/add-profile-picture/', ProfilePictureUploadView.as_view(), name="profilepic_upload"),
    path('registration/complete-profile/<int:pk>/', ProfileCompletionView.as_view(), name='profile_completion'),
    path('registration/add-address/', AddressView.as_view(), name='address'),
    path('logout/', LogoutView.as_view(), name='logout'),
]
