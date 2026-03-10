from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import SummaryFeedback


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')


class UploadVideoForm(forms.Form):
    title = forms.CharField(max_length=255)
    video = forms.FileField()


class FeedbackForm(forms.ModelForm):
    class Meta:
        model = SummaryFeedback
        fields = ('rating',)
