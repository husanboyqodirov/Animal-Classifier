from django.db import models

class Classify(models.Model):
    image = models.ImageField(upload_to='test/')
