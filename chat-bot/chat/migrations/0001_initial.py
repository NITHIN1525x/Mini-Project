from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Conversation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("user_id", models.CharField(db_index=True, max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["-updated_at"]},
        ),
        migrations.CreateModel(
            name="Message",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("user_text", models.TextField()),
                ("bot_response", models.TextField()),
                ("intent_tag", models.CharField(max_length=100)),
                ("confidence", models.FloatField(default=0.0)),
                ("uncertainty", models.FloatField(default=0.0)),
                ("is_uncertain", models.BooleanField(default=False)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
                ("conversation", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="messages", to="chat.conversation")),
            ],
            options={"ordering": ["timestamp"]},
        ),
        migrations.CreateModel(
            name="PredictionMetric",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
                ("intent", models.CharField(db_index=True, max_length=100)),
                ("confidence", models.FloatField(default=0.0)),
                ("uncertainty", models.FloatField(default=0.0)),
                ("user_satisfied", models.BooleanField(blank=True, null=True)),
                ("conversation", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="prediction_metrics", to="chat.conversation")),
            ],
            options={"ordering": ["-timestamp"]},
        ),
        migrations.CreateModel(
            name="UserFeedback",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("predicted_intent", models.CharField(max_length=100)),
                ("corrected_intent", models.CharField(blank=True, max_length=100)),
                ("user_rating", models.IntegerField(choices=[(1, "Poor"), (2, "OK"), (3, "Good"), (4, "Excellent")], default=3)),
                ("feedback_text", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("message", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="feedback", to="chat.message")),
            ],
            options={"ordering": ["-created_at"]},
        ),
    ]
