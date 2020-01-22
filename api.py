import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
from skimage import color
import numpy as np
from skimage import io


app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':

        file = request.files['image']

        if not file: return render_template('index.html', label="No file")

        img = misc.imread(file)
        img = color.rgb2gray(img)
        img = np.asarray(img)
        img =img/255.

        #img = img.reshape(img.shape[0],img.shape[1]* img.shape[2])

        prediction = model.predict(img)
        label = str(np.squeeze(prediction))

        if label=='10': label='0'
            
        return render_template('index.html', label=label)


if __name__ == '__main__':
    model = joblib.load('model.pkl')
    app.run()
