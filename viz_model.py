#Visualize model

from keras.utils.visualize_util import plot
from keras.models import model_from_json
import json

model_file = 'steering_angle'
#Open the model at the top level dir
with open(model_file+'.json', 'r') as jfile:
    model = model_from_json(json.load(jfile))

plot(model, to_file='model.png')