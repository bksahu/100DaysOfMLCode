# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

# instantiate flask
app = flask.Flask(__name__)

# we need to redefine our metric function in order
# to use it when loading the model


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom metric function
global graph
graph = tf.get_default_graph()
model = load_model('games.h5', custom_objects={'auc': auc})

# define a predict function as an endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = str(model.predict(x)[0][0])
            data["success"] = True

    # return a response in json format
    return flask.jsonify(data)

# start the flask app, allow remote connections
app.run()

'''Tests
1. Using browser
  ---------------
Goto http://127.0.0.1:5000/predict?g1=1&g2=0&g3=0&g4=0&g5=0&g6=0&g7=0&g8=0&g9=0&g10=0
2. Using Curl
  ------------
$ curl -X POST -H "Content-Type: application/json" -d "{ \"g1\": 1, \"g2\": 0, \"g3\": 0, \"g4\": 0, \"g5\": 0, \"g6\": 0, \"g7\": 0, \"g8\": 0, \"g9\": 0, \"g10\": 0 }" http://127.0.0.1:5000/predict
'''
