from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('analytics/finance/', views.analytics_finance_view, name='analytics_finance'),
    path('analytics/productivity/', views.analytics_productivity_view, name='analytics_productivity'),
    path('analytics/behavior/', views.analytics_behavior_view, name='analytics_behavior'),
    path('simulation/', views.simulation_sandbox_view, name='simulation_sandbox'),
    path('api/simulate/', views.api_run_simulation, name='api_simulate'),
    path('api/chat/', views.api_assistant_chat, name='api_chat'),
    path('entry/daily/', views.manual_daily_entry, name='daily_entry'),
    path('upload/csv/', views.upload_csv_view, name='upload_csv'),
    path('profile/', views.profile_view, name='profile'),
    path('goals/add/', views.add_goal, name='add_goal'),
    path('goals/<int:goal_id>/update/', views.update_goal_progress, name='update_goal_progress'),
    path('goals/<int:goal_id>/delete/', views.delete_goal, name='delete_goal'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]