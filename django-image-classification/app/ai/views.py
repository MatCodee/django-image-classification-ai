from cProfile import label
from django.shortcuts import render

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
import os

HEIGHT = 224
WIDTH = 224

with open('models/imagenet_classes.json','r') as f:
    labelInfo = f.read()

labelInfo = json.loads(labelInfo)

# cargamos el modelo
model = load_model('./models/MobileNetModelImagenet.h5')


model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/MobileNetModelImagenet.h5')


# Create your views here.
def index(request):
    context = {}
    return render(request,'index.html',context)


def predictImage(request):
    if request.method == 'POST':
        object_file = request.FILES['filePath']
        
        fs = FileSystemStorage()
        file_path_name = fs.save(object_file.name,object_file)
        file_path_name = fs.url(file_path_name)


        testimage='.'+file_path_name
        img = image.load_img(testimage, target_size=(HEIGHT, WIDTH))
        x = image.img_to_array(img)
        x=x/255
        x=x.reshape(1,HEIGHT, WIDTH,3)
        with model_graph.as_default():
            with tf_session.as_default():
                predi=model.predict(x)

        import numpy as np
        predictedLabel=labelInfo[str(np.argmax(predi[0]))]
        
        context = {
            'file_path_name': file_path_name,
            'predictedLabel':predictedLabel[1],
        }
        return render(request,'index.html',context)
    
def viewDataBase(request):
    context = {}
    list_img_db = os.listdir('./media/')
    list_img_path = ['./media/'+ i  for i in list_img_db]
    
    context['list_img'] = list_img_path
    return render(request,'viewDB.html',context)