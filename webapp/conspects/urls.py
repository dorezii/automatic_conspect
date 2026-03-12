from django.urls import path

from . import views

urlpatterns = [
    path('', views.public_summaries, name='public_summaries'),
    path('signup/', views.signup_view, name='signup'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_video, name='upload_video'),
    path('summary/<int:summary_id>/', views.summary_detail, name='summary_detail'),
    path('summary/<int:summary_id>/download/', views.download_summary, name='download_summary'),
]
