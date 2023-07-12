from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
from predictFromModel import prediction

os.putenv('LANG','en_US.UTF-8')
os.putenv('LC_ALL','en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/",methods = ['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path) #object initialization

            # prediction for dataset present in database
            path = pred.predictionFromModel()

            return Response("Prediction File created at %s!!!" %path)

        elif request.form is not None:
            path = request.form['filepath']

            pred_val = pred_validation(path)

            pred_val.prediction_validation()

            pred = prediction(path)

            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" %path)

    except ValueError:
        return Response("Error Occured! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)


@app.route("/train", methods = ['Post'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json['filepath'] is not None:
            path = request.json['filepath']
            train_valObj = train_validation(path) #object inititalization

            train_valObj.train_validation() #calling the training_validation

            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table

    except ValueError:

        return Response("Error Occurred! %s" %ValueError)

    except KeyError:

        return Response("Error Occurred! %s" %KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" %e)
    return Response("Training successfull!!")

port = int(os.getenv("PORT",50001))

if __name__ == "__main__":
    app.run(port = port, debug = True)