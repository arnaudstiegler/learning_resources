import base64
from rest_framework import serializers
from documents.models import Document


class Base64ImageFieldSerializer(serializers.Field):
    def to_representation(self, value):
        return base64.b64encode(value).decode('utf-8')

    def to_internal_value(self, data):
        return data.read()


class DocumentSerializer(serializers.ModelSerializer):

    image = Base64ImageFieldSerializer()

    class Meta:
        model = Document
        fields = ('pk', 'name', 'document_type', 'image')
