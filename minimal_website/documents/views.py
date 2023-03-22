from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from documents.models import Document
from documents.serializers import DocumentSerializer
import openai
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


SYSTEM_INITIAL_MESSAGE = {'role': 'system', 'content': 'You are a helpful assistant. Be succinct for all your answers.'}


class Roles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    is_user: bool
    message: str

    def to_dict(self):
        return {'role': Roles.USER.value if self.is_user else Roles.ASSISTANT.value, 'content': self.message}


def format_messages(data: Any):
    messages = [SYSTEM_INITIAL_MESSAGE]
    for message in data:
        messages.append(Message(message['is_user'], message['message']).to_dict())
    return messages


@api_view(['POST'])
def chat_api_request(request):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages = request.data['messages']
    messages = format_messages(messages)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return Response({'message': completion.choices[0].message["content"].strip()})


@api_view(['GET', 'POST'])
def documents_list(request):
    if request.method == 'GET':
        documents = Document.objects.all()
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        serializer = DocumentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT', 'DELETE'])
def documents_detail(request, pk):
    try:
        document = Document.objects.get(pk=pk)
    except Document.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'PUT':
        serializer = DocumentSerializer(document, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    elif request.method == 'DELETE':
        document.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)