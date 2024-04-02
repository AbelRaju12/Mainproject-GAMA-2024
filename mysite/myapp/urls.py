from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about', views.about, name='about'),
    path('hospitallocator', views.hospitallocator, name='hospitallocator'),
    path('hospital_table_view', views.hospital_table_view, name='hospital_table_view'),
    path('herb_detection_page', views.herb_detection_page, name='herb_detection_page'),
    path('medicine_data_page', views.medicine_data_page, name='medicine_data_page'),
]