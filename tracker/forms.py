from django import forms
from django.contrib.auth.models import User
from .models import UserProfile, DailyTelemetryRecord

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:border-cyan-500 text-sm',
        'placeholder': '••••••••'
    }))
    password_confirm = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:border-cyan-500 text-sm',
        'placeholder': '••••••••'
    }))

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:border-cyan-500 text-sm', 'placeholder': 'alex_mercer'}),
            'first_name': forms.TextInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:border-cyan-500 text-sm', 'placeholder': 'Alex'}),
            'last_name': forms.TextInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:border-cyan-500 text-sm', 'placeholder': 'Mercer'}),
            'email': forms.EmailInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-3 focus:outline-none focus:border-cyan-500 text-sm', 'placeholder': 'alex@example.com'}),
        }

    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get("password") != cleaned_data.get("password_confirm"):
            raise forms.ValidationError("Passwords do not match!")
        return cleaned_data

from django import forms
from django.contrib.auth.models import User
from .models import UserProfile, DailyTelemetryRecord

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['profession', 'age', 'bio', 'monthly_income', 'monthly_budget', 'target_study_hours_weekly']
        widgets = {
            'profession': forms.TextInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-2.5 text-xs'}),
            'age': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-2.5 text-xs'}),
            'bio': forms.Textarea(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-2.5 text-xs', 'rows': 2}),
            'monthly_income': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-2.5 text-xs'}),
            'monthly_budget': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-2.5 text-xs'}),
            'target_study_hours_weekly': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-4 py-2.5 text-xs'}),
        }

class DailyTelemetryManualForm(forms.ModelForm):
    class Meta:
        model = DailyTelemetryRecord
        exclude = ['user', 'total_expenses', 'savings', 'productivity_score', 'behavioral_state', 'next_day_expenses', 'next_day_productivity']
        widgets = {
            'entry_date': forms.DateInput(attrs={'type': 'date', 'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs'}),
            'day_of_week': forms.TextInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs'}),
            'income': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.01', 'value': '0.00'}),
            'food_expenses': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.01'}),
            'travel_expenses': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.01'}),
            'shopping_expenses': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.01'}),
            'utilities_expenses': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.01'}),
            'health_expenses': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.01'}),
            'other_expenses': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.01'}),
            'work_hours': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.1'}),
            'study_hours': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.1'}),
            'break_minutes': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs'}),
            'screen_time_hours': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.1'}),
            'tasks_assigned': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs'}),
            'tasks_completed': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs'}),
            'steps_count': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs'}),
            'exercise_duration_min': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs'}),
            'sleep_hours': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.1'}),
            'reading_meditation_min': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs'}),
            'habit_completion_pct': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'step': '0.1'}),
            'stress_level': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'min': 1, 'max': 10}),
            'mood_score': forms.NumberInput(attrs={'class': 'w-full bg-slate-900 border border-slate-700 text-white rounded-xl px-3 py-2 text-xs', 'min': 1, 'max': 10}),
        }