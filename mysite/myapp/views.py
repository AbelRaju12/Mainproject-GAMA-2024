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
                     'Doddpathre',
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
                     'Spinach1',
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

        model = tf.keras.models.load_model('modelv1')

        img = tf.keras.preprocessing.image.load_img(
            file_name, target_size=(299, 299)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.sigmoid(predictions[0])

        image_url = "/static/myapp/TestImage.jpg"
        result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(labels[np.argmax(score)], 100 * np.max(score))
        google_search = "https://www.google.com/search?q={}_herb".format(labels[np.argmax(score)])
        
        return render(request, "resultpage.html",context={"result":result, "image_url":image_url, "google_search":google_search})
    return render(request, 'herb_detection_page.html')

def medicine_data_page(request):

    ayurvedic_medicines = [
            {
                "name": "Triphala",
                "description": "Triphala is a combination of three fruits: Amalaki, Bibhitaki, and Haritaki. It is known for its antioxidant properties and is used to promote digestive health, improve immunity, and support detoxification.",
                "treats": "Constipation, indigestion, detoxification"
            },
            {
                "name": "Ashwagandha",
                "description": "Ashwagandha is an adaptogenic herb known for its ability to reduce stress, anxiety, and fatigue. It also helps in improving stamina, boosting immunity, and enhancing overall vitality.",
                "treats": "Stress, anxiety, fatigue, low immunity"
            },
            {
                "name": "Turmeric",
                "description": "Turmeric, or Curcuma longa, is a spice with powerful anti-inflammatory and antioxidant properties. It is commonly used in Ayurveda to support joint health, improve digestion, and promote radiant skin.",
                "treats": "Inflammation, joint pain, digestive issues"
            },
            {
                "name": "Neem",
                "description": "Neem, or Azadirachta indica, is a medicinal plant with antibacterial, antiviral, and antifungal properties. It is used to treat various skin conditions, promote oral health, and support detoxification.",
                "treats": "Skin infections, oral hygiene, detoxification"
            },
            {
                "name": "Guggul",
                "description": "Guggul is a resin obtained from the Commiphora mukul tree. It is known for its cholesterol-lowering properties and is used to support cardiovascular health, weight management, and thyroid function.",
                "treats": "High cholesterol, obesity, thyroid disorders"
            },
            {
                "name": "Brahmi",
                "description": "Brahmi, or Bacopa monnieri, is a herb known for its cognitive-enhancing properties. It is used to improve memory, concentration, and overall brain function. Brahmi is also beneficial for reducing stress and anxiety.",
                "treats": "Memory loss, poor concentration, stress"
            },
            {
                "name": "Trikatu",
                "description": "Trikatu is a blend of three warming spices: ginger, black pepper, and long pepper. It aids digestion, enhances metabolism, and helps in weight management. Trikatu is also beneficial for respiratory health.",
                "treats": "Indigestion, sluggish metabolism, respiratory issues"
            },
            {
                "name": "Shatavari",
                "description": "Shatavari, or Asparagus racemosus, is a rejuvenating herb known for its hormone-balancing properties. It supports female reproductive health, relieves menstrual discomfort, and promotes lactation. Shatavari also has anti-inflammatory and antioxidant effects.",
                "treats": "Menstrual disorders, menopausal symptoms, lactation support"
            },
            {
                "name": "Arjuna",
                "description": "Arjuna, or Terminalia arjuna, is a tree bark with cardio-protective properties. It strengthens the heart muscles, improves circulation, and helps in managing hypertension. Arjuna also has antioxidant and anti-inflammatory effects.",
                "treats": "Heart problems, hypertension, poor circulation"
            },
            {
                "name": "Tulsi",
                "description": "Tulsi, or Holy Basil, is a sacred herb known for its medicinal properties. It boosts immunity, relieves respiratory conditions, and supports stress management. Tulsi also has antimicrobial and anti-inflammatory effects.",
                "treats": "Cough, cold, respiratory infections, stress"
            },
            {
                "name": "Guduchi",
                "description": "Guduchi, or Tinospora cordifolia, is an immune-boosting herb with detoxifying properties. It helps in managing fever, enhances immunity, and promotes overall health and vitality.",
                "treats": "Fever, low immunity, general debility"
            },
            {
                "name": "Haritaki",
                "description": "Haritaki, or Terminalia chebula, is one of the three fruits in Triphala. It is known for its rejuvenating and detoxifying properties. Haritaki supports digestive health, improves liver function, and enhances immunity.",
                "treats": "Digestive issues, liver disorders, immunity support"
            },
            {
                "name": "Shilajit",
                "description": "Shilajit is a sticky substance found in the rocks of the Himalayas. It is rich in fulvic acid and minerals, making it a potent rejuvenating tonic. Shilajit boosts energy levels, enhances vitality, and promotes overall wellness.",
                "treats": "Fatigue, weakness, low vitality"
            },
            {
                "name": "Punarnava",
                "description": "Punarnava, or Boerhavia diffusa, is a diuretic herb used to manage kidney disorders and fluid retention. It helps in reducing swelling, improving urine flow, and supporting kidney function. Punarnava also has anti-inflammatory effects.",
                "treats": "Edema, kidney problems, urinary tract infections"
            },
            {
                "name": "Amalaki",
                "description": "Amalaki, or Indian gooseberry, is one of the fruits in Triphala. It is rich in vitamin C and antioxidants, making it a potent rejuvenating herb. Amalaki supports immune function, improves digestion, and enhances skin health.",
                "treats": "Weak immunity, digestive issues, skin problems"
            },
            {
                "name": "Musta",
                "description": "Musta, or Cyperus rotundus, is a herb known for its digestive and carminative properties. It helps in relieving indigestion, bloating, and abdominal discomfort. Musta also has antimicrobial effects.",
                "treats": "Indigestion, flatulence, abdominal pain"
            },
            {
                "name": "Vasaka",
                "description": "Vasaka, or Adhatoda vasica, is a medicinal plant used in respiratory disorders. It acts as a bronchodilator, helping to relieve cough, asthma, and bronchitis. Vasaka also has expectorant and anti-inflammatory effects.",
                "treats": "Cough, asthma, bronchitis"
            },
            {
                "name": "Kutki",
                "description": "Kutki, or Picrorhiza kurroa, is a bitter herb used for liver disorders and immune support. It stimulates bile production, improves liver function, and helps in detoxification. Kutki also has anti-inflammatory properties.",
                "treats": "Liver disorders, jaundice, immune support"
            },
            {
                "name": "Bhringraj",
                "description": "Bhringraj, or Eclipta alba, is a herb known for its hair-rejuvenating properties. It strengthens hair roots, prevents hair loss, and promotes hair growth. Bhringraj also has anti-inflammatory effects.",
                "treats": "Hair loss, premature graying, scalp conditions"
            },
            {
                "name": "Yashtimadhu",
                "description": "Yashtimadhu, or Licorice, is a sweet-tasting herb used in various Ayurvedic formulations. It soothes the digestive system, relieves cough and sore throat, and supports adrenal function. Yashtimadhu also has anti-inflammatory and immune-modulating effects.",
                "treats": "Acid reflux, cough, adrenal support"
            }
        ]

    
    return render(request, 'medicine_data_page.html',{'ayurvedic_medicines': ayurvedic_medicines})