from django.contrib import admin

from .models import Summary, SummaryFeedback


@admin.register(Summary)
class SummaryAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user', 'created_at', 'downloads', 'visits')


@admin.register(SummaryFeedback)
class SummaryFeedbackAdmin(admin.ModelAdmin):
    list_display = ('id', 'summary', 'user', 'rating', 'created_at')
