from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import User
from django.views.generic import CreateView, UpdateView

from .forms import *
from .models import Profile, Address

# Create your views here.

class UserRegistrationView(CreateView):
    model = User
    form_class = UserRegistrationForm

    template_name = "registration/registration.html"

    def form_valid(self, form):
        if form.cleaned_data['password'] == form.cleaned_data['confirm_password']:
            user = form.save(commit=False)
            user.set_password(user.password)
            user.save()
            new_user = authenticate(
                            username=form.cleaned_data['username'],
                            password=form.cleaned_data['password'],
                        )
            login(self.request, new_user)
            return redirect('/profile/registration/add-profile-picture')
        else:
            return super().form_valid(form)

class ProfilePictureUploadView(LoginRequiredMixin, CreateView):
    model = Profile
    form_class = ProfilePictureForm

    template_name = "registration/profilepic_upload.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        form.save()
        return redirect('/profile/registration/complete-profile/'+str(self.request.user.profile.id))

class ProfileCompletionView(UpdateView):
    model = Profile
    form_class = ProfileCompletionForm

    template_name = "registration/profile_complete.html"

    def form_valid(self, form):
        form.save()
        return redirect('/profile/registration/add-address/')

class AddressView(LoginRequiredMixin, CreateView):
    model = Address
    form_class = AddressForm

    template_name = "registration/address.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        form.save()
        return redirect('')