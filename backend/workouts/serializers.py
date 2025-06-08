from rest_framework import serializers
from .models import WorkoutLog


class WorkoutLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkoutLog
        fields = ('id', 'user', 'raw_input', 'parsed_summary', 'created_at')
        read_only_fields = ('user', 'parsed_summary', 'created_at')
