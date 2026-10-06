from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

# 1. User Profile & Digital Twin Configuration
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    profession = models.CharField(max_length=120, default='Software Engineer')
    age = models.PositiveIntegerField(default=24)
    bio = models.TextField(blank=True, default='Digital Twin user exploring personal simulation.')
    monthly_income = models.DecimalField(max_digits=12, decimal_places=2, default=85000.00) # ₹ Monthly Salary
    monthly_budget = models.DecimalField(max_digits=12, decimal_places=2, default=50000.00) # ₹ Monthly Budget
    target_study_hours_weekly = models.DecimalField(max_digits=5, decimal_places=1, default=20.0)
    goal_score = models.IntegerField(default=85)
    days_active = models.IntegerField(default=1)

    def __str__(self):
        return f"Profile of {self.user.username}"

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()

# 2. Dynamic Target Objectives
class Goal(models.Model):
    CATEGORY_CHOICES = [
        ('Finance', 'Financial Objective'),
        ('Study', 'Academic / Skill Objective'),
        ('Habit', 'Lifestyle & Health Objective'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='goals')
    title = models.CharField(max_length=150)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='Finance')
    current_value = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    target_value = models.DecimalField(max_digits=12, decimal_places=2, default=100.00)
    unit = models.CharField(max_length=20, default='₹')
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def progress_percentage(self):
        if self.target_value > 0:
            return min(int((self.current_value / self.target_value) * 100), 100)
        return 0

# 3. Preserved Milestone 1 Models (Prevents PostgreSQL table deletion)
class FinancialRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='financial_records')
    category = models.CharField(max_length=100)
    description = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    recurring_frequency = models.CharField(max_length=20, default='Monthly')
    impact_level = models.CharField(max_length=20, default='Medium')
    created_at = models.DateTimeField(auto_now_add=True)

class StudyRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='study_records')
    subject = models.CharField(max_length=120)
    study_hours = models.DecimalField(max_digits=5, decimal_places=2)
    focus_rating = models.IntegerField(default=8)
    key_takeaways = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class HabitRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='habit_records')
    habit_type = models.CharField(max_length=50)
    metric_value = models.CharField(max_length=100)
    consistency_score = models.IntegerField(default=80)
    created_at = models.DateTimeField(auto_now_add=True)

# 4. Consolidated Daily Telemetry Record (Milestone 2 Schema Parity)
class DailyTelemetryRecord(models.Model):
    BEHAVIORAL_STATES = [
        ('Peak Focus', 'Peak Focus'),
        ('Burnout Risk', 'Burnout Risk'),
        ('Distracted', 'Distracted'),
        ('Recovery', 'Recovery'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='telemetry_records')
    entry_date = models.DateField(db_index=True)
    day_of_week = models.CharField(max_length=20)

    # Financial Domain (₹ INR)
    income = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    food_expenses = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    travel_expenses = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    shopping_expenses = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    utilities_expenses = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    health_expenses = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    other_expenses = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    total_expenses = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    savings = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)

    # Productivity Domain
    work_hours = models.DecimalField(max_digits=4, decimal_places=2, default=0.00)
    study_hours = models.DecimalField(max_digits=4, decimal_places=2, default=0.00)
    break_minutes = models.PositiveIntegerField(default=0)
    screen_time_hours = models.DecimalField(max_digits=4, decimal_places=2, default=0.00)
    tasks_assigned = models.PositiveIntegerField(default=1)
    tasks_completed = models.PositiveIntegerField(default=0)
    productivity_score = models.DecimalField(max_digits=5, decimal_places=2, default=0.00)

    # Behavioral & Wellness Domain
    steps_count = models.PositiveIntegerField(default=5000)
    exercise_duration_min = models.PositiveIntegerField(default=0)
    sleep_hours = models.DecimalField(max_digits=4, decimal_places=2, default=7.00)
    reading_meditation_min = models.PositiveIntegerField(default=0)
    habit_completion_pct = models.DecimalField(max_digits=5, decimal_places=2, default=0.00)
    stress_level = models.PositiveIntegerField(default=5)
    mood_score = models.PositiveIntegerField(default=7)
    behavioral_state = models.CharField(max_length=30, choices=BEHAVIORAL_STATES, default='Recovery')

    # Supervised Targets
    next_day_expenses = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    next_day_productivity = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    class Meta:
        ordering = ['-entry_date']
        unique_together = ('user', 'entry_date')

    def __str__(self):
        return f"{self.user.username} - {self.entry_date}"

# 5. Activity Log
class ActivityLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activity_logs')
    category = models.CharField(max_length=50)
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']