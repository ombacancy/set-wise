from django.db import models
from django.conf import settings


class WorkoutLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='workout_logs')
    raw_input = models.TextField()
    parsed_summary = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Workout by {self.user.username} on {self.created_at.strftime('%Y-%m-%d')}"
