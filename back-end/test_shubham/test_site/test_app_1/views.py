from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import base64
import PIL.Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from test_site import settings
from tensorflow.compat.v1.keras.backend import set_session
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import datetime
import traceback


# Create your views here.

def index(request):
    return render(request, 'index.html')

def upload(request):
    if request.method == 'POST':
        f = request.FILES['sentFile']
        response = {}
        filename = 'pic.jpg'
        file_name_2 = default_storage.save(filename, f)
        file_url = default_storage.url(file_name_2)
        original = load_img(file_url, target_size=(512,512))
        numpy_image = img_to_array(original).astype('float')/255
        numpy_image = np.expand_dims(numpy_image,axis=0)
        labels = ['Healthy','Multiple_Diseases','Rust','Scab']
        with settings.GRAPH1.as_default():
            set_session(settings.SESS)
            predictions = settings.model.predict(numpy_image)
        
        label = labels[np.argmax(predictions)]
        response['name'] = str(label)

        return render(request, 'upload.html', response)

    else:
        return render(request, 'upload.html')