from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from googleplaces import GooglePlaces, lang
from django.http import JsonResponse

import numpy as np
from PIL import Image
import tensorflow as tf
import os
from django.templatetags.static import static
from pathlib import Path

from django.shortcuts import render
from myapp.models import AyurvedicMedicine
from .populate import populate_medicines

BASE_DIR = Path(__file__).resolve().parent.parent

API_KEY = ''

def home(request):
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def hospitallocator(request):
    if request.method == "POST":
        userinput = request.POST.get("userinput")

        def search_hospitals_by_area(area_name, api_key):
            google_places = GooglePlaces(api_key)
            query_result = google_places.text_search(
                query='ayurvedic hospitals in ' + area_name,
                language=lang.ENGLISH
            )
            hospitals = []
            for place in query_result.places:
                place.get_details()  # Explicit call to fetch details
                contact = place.local_phone_number if hasattr(place, 'local_phone_number') else "Not available"
                
                hospital_info = {
                    'name': place.name,
                    'contact': contact,
                    'latitude': place.geo_location['lat'],
                    'longitude': place.geo_location['lng']
                }
                hospitals.append(hospital_info)
                
            return hospitals

        hospitals = search_hospitals_by_area(userinput, API_KEY)

        print(hospitals)

        return render(request, 'hospital_table_view.html', {'hospitals': hospitals})



    return render(request, "hospitallocator.html")

def hospital_table_view(request):
    
    return render(request, 'hospital_table_view.html')

def herb_ml(request):
    return render(request, 'herb_ml.html')


from .herb_description import labels_info

def herb_detection_page(request):
    if request.method == "POST":
        my_uploaded_file = request.FILES['my_uploaded_file'].read()

        labels = ['Aloevera',
                     'Amla',
                     'Amruthaballi',
                     'Arali',
                     'Astma_weed',
                     'Badipala',
                     'Balloon_Vine',
                     'Bamboo',
                     'Beans',
                     'Betel',
                     'Bhrami',
                     'Bringaraja',
                     'Caricature',
                     'Castor',
                     'Catharanthus',
                     'Chakte',
                     'Chilly',
                     'Citron lime (herelikai)',
                     'Coffee',
                     'Common rue(naagdalli)',
                     'Coriender',
                     'Curry',
                     'Drumstick',
                     'Ekka',
                     'Eucalyptus',
                     'Ganigale',
                     'Ganike',
                     'Gasagase',
                     'Ginger',
                     'Globe Amarnath',
                     'Guava',
                     'Henna',
                     'Hibiscus',
                     'Honge',
                     'Indian_Borage',
                     'Insulin',
                     'Jackfruit',
                     'Jasmine',
                     'Kambajala',
                     'Kasambruga',
                     'Kohlrabi',
                     'Lantana',
                     'Lemon',
                     'Lemongrass',
                     'Malabar_Nut',
                     'Malabar_Spinach',
                     'Mango',
                     'Marigold',
                     'Mint',
                     'Neem',
                     'Nelavembu',
                     'Nerale',
                     'Nooni',
                     'Onion',
                     'Padri',
                     'Palak(Spinach)',
                     'Papaya',
                     'Parijatha',
                     'Pea',
                     'Pepper',
                     'Pomoegranate',
                     'Pumpkin',
                     'Raddish',
                     'Rose',
                     'Sampige',
                     'Sapota',
                     'Seethaashoka',
                     'Seethapala',
                     'Spinach',
                     'Tamarind',
                     'Taro',
                     'Tecoma',
                     'Thumbe',
                     'Tomato',
                     'Tulsi',
                     'Turmeric',
                     'ashoka',
                     'camphor',
                     'kamakasturi',
                     'kepala']

        file_name = "{}{}.jpg".format(os.path.join(BASE_DIR, 'myapp/static/myapp/'),"TestImage")
            
        with open(file_name,'wb') as f:
            f.write(my_uploaded_file)

        model = tf.keras.models.load_model('modelv1.h5')

        img = tf.keras.preprocessing.image.load_img(
            file_name, target_size=(299, 299)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.sigmoid(predictions[0])

        image_url = "/static/myapp/TestImage.jpg"
        herb_label = labels[np.argmax(score)]
        result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(labels[np.argmax(score)], 100 * np.max(score))
        google_search = "https://www.google.com/search?q={}%20herb".format(labels[np.argmax(score)])
        
        # Fetching herb information
        herb_description = labels_info[herb_label]['description']
        scientific_name = labels_info[herb_label]['scientific_name']
        uses = labels_info[herb_label]['uses']

        return render(request, "resultpage.html", context={
            "result": result, 
            "image_url": image_url, 
            "herb_description": herb_description,
            "scientific_name": scientific_name,
            "uses": uses,
            "google_search": google_search
        })
    return render(request, 'herb_detection_page.html')

def herb_detection_page2(request):
    if request.method == "POST":
        my_uploaded_file = request.FILES['my_uploaded_file'].read()

        labels = ['Aloevera',
                  'Amla',
                  'Amruthaballi',
                  'Arali',
                  'Ashoka',
                  'Ashwagandha',
                  'Avacado',
                  'Bamboo',
                  'Basale',
                  'Betel',
                  'Betel_Nut',
                  'Brahmi',
                  'Castor',
                  'Curry',
                  'Ekka',
                  'Ganike',
                  'Guava',
                  'Geranium',
                  'Henna',
                  'Hibiscus',
                  'Honge',
                  'Indian_Borage',
                  'Insulin',
                  'Jasmine',
                  'Lemon',
                  'Lemon_grass',
                  'Mango',
                  'Mint',
                  'Nagadali',
                  'Neem',
                  'Nithyapushpa',
                  'Nooni',
                  'Pappaya',
                  'Pepper',
                  'Pomegranate',
                  'Raktachandini',
                  'Rose',
                  'Sapota',
                  'Tulsi', 
                  'Wood_sorel']

        file_name = "{}{}.jpg".format(os.path.join(BASE_DIR, 'myapp/static/myapp/'),"TestImage")
            
        with open(file_name,'wb') as f:
            f.write(my_uploaded_file)

        model = tf.keras.models.load_model('modelv2.h5')

        img = tf.keras.preprocessing.image.load_img(
            file_name, target_size=(299, 299)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.sigmoid(predictions[0])

        image_url = "/static/myapp/TestImage.jpg"
        herb_label = labels[np.argmax(score)]
        result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(labels[np.argmax(score)], 100 * np.max(score))
        google_search = "https://www.google.com/search?q={}%20herb".format(labels[np.argmax(score)])
        
        # Fetching herb information
        herb_description = labels_info[herb_label]['description']
        scientific_name = labels_info[herb_label]['scientific_name']
        uses = labels_info[herb_label]['uses']

        return render(request, "resultpage.html", context={
            "result": result, 
            "image_url": image_url, 
            "herb_description": herb_description,
            "scientific_name": scientific_name,
            "uses": uses,
            "google_search": google_search
        })
    return render(request, 'herb_detection_page2.html')

def medicine_data_page(request):
    # Uncomment the line below if you want to populate the medicines when this view is called
    # populate_medicines()
    # Delete all the objects in the AyurvedicMedicine table
    # AyurvedicMedicine.objects.all().delete()
    ayurvedic_medicines = AyurvedicMedicine.objects.all().order_by('name')
    return render(request, 'medicine_data_page.html', {'ayurvedic_medicines': ayurvedic_medicines})