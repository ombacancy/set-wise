from rest_framework import viewsets, permissions
from .models import WorkoutLog
from .serializers import WorkoutLogSerializer
from .llm_client import FastAPILLMClient


class WorkoutLogViewSet(viewsets.ModelViewSet):
    serializer_class = WorkoutLogSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return WorkoutLog.objects.filter(user=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        raw_text = serializer.validated_data['raw_input']
        llm_client = FastAPILLMClient()
        parsed_summary = llm_client.parse_workout(raw_text)
        serializer.save(user=self.request.user, parsed_summary=parsed_summary)
