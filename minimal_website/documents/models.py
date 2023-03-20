from django.db import models


class Document(models.Model):
    name = models.CharField(max_length=240)
    document_type = models.CharField(max_length=120)
    image = models.BinaryField()

    def __str__(self):
        return self.name
