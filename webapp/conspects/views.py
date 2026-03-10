from pathlib import Path
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.db.models import Avg
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .forms import FeedbackForm, SignUpForm, UploadVideoForm
from .models import Summary, SummaryFeedback
from .pipeline import generate_summary_from_video


def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form})


@login_required
def dashboard(request):
    items = Summary.objects.filter(user=request.user).annotate(avg_rating=Avg('feedbacks__rating'))
    return render(request, 'conspects/dashboard.html', {'items': items})


@login_required
def upload_video(request):
    if request.method == 'POST':
        form = UploadVideoForm(request.POST, request.FILES)
        if form.is_valid():
            start_day = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if Summary.objects.filter(user=request.user, created_at__gte=start_day).count() >= 1:
                messages.error(request, 'Лимит достигнут: можно создавать только 1 конспект в день.')
                return redirect('dashboard')

            video = form.cleaned_data['video']
            title = form.cleaned_data['title']
            summary_text = generate_summary_from_video(Path(video.name))
            summary = Summary.objects.create(
                user=request.user,
                title=title,
                video=video,
                summary_text=summary_text,
            )
            send_mail(
                subject=f'Ваш конспект готов: {summary.title}',
                message=summary.summary_text,
                from_email=None,
                recipient_list=[request.user.email],
                fail_silently=True,
            )
            summary.email_sent = True
            summary.save(update_fields=['email_sent'])
            messages.success(request, 'Конспект создан и отправлен на почту (если почта настроена).')
            return redirect('dashboard')
    else:
        form = UploadVideoForm()

    return render(request, 'conspects/upload.html', {'form': form})


def public_summaries(request):
    items = Summary.objects.annotate(avg_rating=Avg('feedbacks__rating')).order_by('-downloads', '-visits', '-created_at')
    return render(request, 'conspects/public_list.html', {'items': items})


def summary_detail(request, summary_id):
    item = get_object_or_404(Summary.objects.annotate(avg_rating=Avg('feedbacks__rating')), pk=summary_id)
    Summary.objects.filter(pk=item.pk).update(visits=item.visits + 1)

    form = FeedbackForm()
    if request.method == 'POST' and request.user.is_authenticated:
        form = FeedbackForm(request.POST)
        if form.is_valid():
            SummaryFeedback.objects.update_or_create(
                user=request.user,
                summary=item,
                defaults={'rating': form.cleaned_data['rating']},
            )
            messages.success(request, 'Спасибо за отзыв!')
            return redirect('summary_detail', summary_id=item.id)

    return render(request, 'conspects/detail.html', {'item': item, 'form': form})


def download_summary(request, summary_id):
    item = get_object_or_404(Summary, pk=summary_id)
    Summary.objects.filter(pk=item.pk).update(downloads=item.downloads + 1)
    response = HttpResponse(item.summary_text, content_type='text/plain; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="summary_{item.id}.txt"'
    return response
