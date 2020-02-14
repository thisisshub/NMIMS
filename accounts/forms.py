from django import forms
from django.contrib.auth.models import User

from .models import Profile, Address

class UserRegistrationForm(forms.ModelForm):

    first_name = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'First Name'}) , required=True)
    last_name = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'Last Name'}) , required=True)
    username = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'Username'}) , required=True)
    email = forms.EmailField(widget=forms.EmailInput(attrs={'placeholder':'Email'}), required=True)
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder':'Password'}))
    confirm_password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder':'Confirm Password'}))
    
    class Meta:
        model = User
        fields = (
            'first_name',
            'last_name',
            'username',
            'email',
            'password',
        )
    
    def clean(self):
        cleaned_data = self.cleaned_data
        passw = cleaned_data.get('password')
        c_pass = cleaned_data.get('confirm_password')

        if c_pass != passw:
            self.add_error('confirm_password', "Password didn't matched")
        return cleaned_data

class ProfilePictureForm(forms.ModelForm):
    profilepic = forms.ImageField(required=False)
    
    class Meta:
        model = Profile
        fields = ('profilepic',)

class ProfileCompletionForm(forms.ModelForm):

    profession = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'Profession'}), required=False)
    bio = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'Bio'}), required=False)
    phone = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'Phone'}), required=False)
    
    class Meta:
        model = Profile
        fields = (
            'profession',
            'bio',
            'phone',
        )

class AddressForm(forms.ModelForm):
    
    house_number = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'House Number'}) , required=True)
    locality = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'Locality'}) , required=True)
    landmark = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'Landmark'}) , required=True)
    city = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'City'}) , required=True)
    state = forms.CharField(widget=forms.TextInput(attrs={'placeholder':'State'}) , required=True)
    class Meta:
        model = Address
        fields = (
            'house_number',
            'locality',
            'landmark',
            'city',
            'state'
        )