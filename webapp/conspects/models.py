from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Summary(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='summaries')
    title = models.CharField(max_length=255)
    video = models.FileField(upload_to='videos/')
    summary_text = models.TextField()
    email_sent = models.BooleanField(default=False)
    visits = models.PositiveIntegerField(default=0)
    downloads = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title

    @property
    def relevance_score(self):
        return self.visits + (self.downloads * 2)


class SummaryFeedback(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    summary = models.ForeignKey(Summary, on_delete=models.CASCADE, related_name='feedbacks')
    rating = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'summary')

    def __str__(self):
        return f'{self.summary_id}: {self.rating}'
