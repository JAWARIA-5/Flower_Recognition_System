from flask import Flask, render_template, request, jsonify
from predict import predict
import json
import pickle

app = Flask(__name__)

with open('model_pickle.pkl', 'rb') as f:
    model = pickle.load(f)
with open('class_to_name.json', 'r') as f:
    class_to_name = json.load(f)

@app.route('/', methods=['GET'])

def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_flower():
    imagefile = request.files['imagefile']
    imagepath = "./images/" + imagefile.filename
    imagefile.save(imagepath)
    probs, classes = predict(imagepath, model, topk=1)
    class_name = class_to_name[classes[0]]
    probability = probs[0]
    imagepathforhtml = "../images/" + imagefile.filename
    return render_template('recognition.html', flower_name=class_name, probability=probability)

if __name__ == '__main__':
    app.run(debug=False)
