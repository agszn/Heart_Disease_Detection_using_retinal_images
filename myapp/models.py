from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User

class Consumer(models.Model):
    name = models.CharField(max_length=100, verbose_name="Student Name")
    email = models.EmailField(max_length=277, verbose_name="Student Email")
    image = models.ImageField(upload_to='consumer_images/', null=True, blank=True, verbose_name="Consumer Image")
    content = models.TextField(verbose_name="Consumer Content", blank=True, null=True)

    def __str__(self):
        return str(self.id)

from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)

def create_profile(sender, **kwargs):
    if kwargs['created']:
        Profile.objects.create(user=kwargs['instance'])

models.signals.post_save.connect(create_profile, sender=User)

class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()

    def __str__(self):
        return self.name

class PredictionResult(models.Model):
    age = models.FloatField()
    sbp = models.FloatField()  # Systolic Blood Pressure
    dbp = models.FloatField()  # Diastolic Blood Pressure
    bmi = models.FloatField()  # Body Mass Index
    hba1c = models.FloatField()  # HbA1c (Blood Sugar)
    
    true_age = models.FloatField(null=True)
    true_sbp = models.FloatField(null=True)
    true_dbp = models.FloatField(null=True)
    true_bmi = models.FloatField(null=True)
    true_hba1c = models.FloatField(default=0.0)  # Add a default value for true_hba1c

    mae = models.FloatField(default=0.0)  # Mean Absolute Error
    
    risk_status = models.CharField(max_length=20, null=True)  # e.g., "At Risk" or "Healthy"
    created_at = models.DateTimeField(auto_now_add=True, null=True )  # Timestamp when the record is created

    def __str__(self):
        return f"Prediction at {self.created_at} - {self.risk_status}"

