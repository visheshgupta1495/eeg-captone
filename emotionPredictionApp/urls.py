# todos/urls.py
from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.home, name='home'),
    # path('grayscale', views.convertGrayScale, name='convertGrayScale'),
    # path('compress', views.CompressVideo, name='CompressVideo'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)   # FOR IMAGES
