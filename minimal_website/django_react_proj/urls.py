from django.contrib import admin
from django.urls import path, re_path
from documents.views import documents_detail, documents_list

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^api/documents/$', documents_list),
    re_path(r'^api/documents/([0-9])$', documents_detail),
]