import joblib
from flask import Flask, request, json, jsonify, render_template
from werkzeug.exceptions import HTTPException

MODEL_PATH = "models/model.joblib"

app = Flask(__name__)


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors (which is the basic error
    response with Flask).
    """
    # Start with the correct headers and status code from the error
    response = e.get_response()
    # Replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


class MissingKeyError(HTTPException):
    # We can define our own error for the missing key
    code = 422
    name = "Missing key error"
    description = "JSON content missing key 'input'."


class MissingJSON(HTTPException):
    # We can define our own error for missing JSON
    code = 400
    name = "Missing JSON"
    description = "Missing JSON."


class BadInputType(HTTPException):
    # We can define our own error for missing JSON
    code = 425
    name = "Input_type error"
    description = "the input must be a list of list"


class BadLenList(HTTPException):
    # We can define our own error for missing JSON
    code = 430
    name = "Bad len input"
    description = "the input shape must be (nb_predictions, 11) "


class BadType(HTTPException):
    # We can define our own error for missing JSON
    code = 435
    name = "List input_Type error"
    description = "the input must be list of floats "

def make_prediction(input):
    """Return a prediction with our regression model.
    """
    # Load model
    regressor = joblib.load(MODEL_PATH)
    # Make prediction (the regressor expects a 2D array that is why we put year
    # in a list of list) and return it
    prediction = regressor.predict(input)
    if prediction.shape[0]==1:
        return prediction[0]
    else:
        return prediction.tolist()


def good_format(input):
    try:
        return sum([isinstance(i,list) for i in input])==len(input)
    except :
        return False

def is_num(x):
    return isinstance(x,float) or isinstance(x,int)


@app.route("/predict", methods=["POST"])
def predict():
    # Check parameters
    if request.json:
        # Get JSON as dictionnary
        json_input = request.get_json()
        if "input" not in json_input:
            # If 'input' is not in our JSON we raise our own error
            raise MissingKeyError()
        
        input_list=json_input["input"]
        # check the input and call our predict function that handle loading model and making a        
        if good_format(input_list):
            if sum([len(i)==11 for i in input_list])!=len(input_list):
                raise BadLenList()
                
            elif sum([sum([is_num(i) for i in x])==11 for x in input_list])!=len(input_list):
                raise BadType()

            else : 
                # prediction
                #print(json_input["input"])
                prediction = make_prediction(input_list)
                # Return prediction
                response = {
                    # Since prediction is a float and jsonify function can't handle
                    # floats we need to convert it to string
                    "prediction": str(prediction),
                }
                return jsonify(response), 200

        else : 
            raise BadInputType()

    raise MissingJSON()


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
