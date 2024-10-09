from django import forms
from django.contrib.auth.models import User
from .models import *

class LoginForm(forms.Form):
    username = forms.CharField(
        max_length=150,
        strip=False,
        required=True,
        help_text='',
        error_messages={
            'unique': "A user with that username already exists.",
        },
    )
    password = forms.CharField(widget=forms.PasswordInput)


class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(
        label='Password', 
        widget=forms.PasswordInput,
        strip=False,
        required=True,
        # help_text="Your password must contain at least 8 characters.",
        error_messages={
            'password_mismatch': "The two password fields didn't match.",
        },
        )
    password2 = forms.CharField(label='Repeat Password', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email')

    def clean_password2(self):
        cd = self.cleaned_data
        if cd['password'] != cd['password']:
            raise forms.ValidationError('Password does not match')
        return cd['password2']

class EditProfileForm(forms.ModelForm):
    profile_picture = forms.ImageField(required=False)
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'profile_picture')


class ContactForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = ['name', 'email', 'message']


from django import forms
class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control image-upload',  # Add Bootstrap class + custom class
            'accept': 'image/*',                   # Accept only image files
            'placeholder': 'Upload Retinal Image',  # Placeholder text
        })
    )


from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Select a retinal image (.tif)')
