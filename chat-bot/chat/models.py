from django.db import models

class Conversation(models.Model):
    user_id = models.CharField(max_length=255, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"Conversation {self.pk} ({self.user_id})"


class Message(models.Model):
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    user_text = models.TextField()
    bot_response = models.TextField()
    intent_tag = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    uncertainty = models.FloatField(default=0.0)
    is_uncertain = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]

    def __str__(self):
        return f"{self.intent_tag} ({self.confidence:.2f})"


class UserFeedback(models.Model):
    RATING_CHOICES = [
        (1, "Poor"),
        (2, "OK"),
        (3, "Good"),
        (4, "Excellent"),
    ]

    message = models.ForeignKey(
        Message,
        on_delete=models.CASCADE,
        related_name="feedback",
    )
    predicted_intent = models.CharField(max_length=100)
    corrected_intent = models.CharField(max_length=100, blank=True)
    user_rating = models.IntegerField(choices=RATING_CHOICES, default=3)
    feedback_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Feedback for message {self.message_id}: {self.user_rating}"


class PredictionMetric(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    intent = models.CharField(max_length=100, db_index=True)
    confidence = models.FloatField(default=0.0)
    uncertainty = models.FloatField(default=0.0)
    user_satisfied = models.BooleanField(null=True, blank=True)
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="prediction_metrics",
    )

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.intent} ({self.confidence:.2f})"
