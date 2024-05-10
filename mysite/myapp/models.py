from django.db import models

class AyurvedicMedicine(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    treats = models.CharField(max_length=200)

    def __str__(self):
        return self.name
