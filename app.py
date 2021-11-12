import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# import model.py file having final model details
import model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_input = request.form['txtarea']

    # First user recommendation

    flag, recom_prods = model.recommendation(user_input)
    if flag == True:
    # Sentiment model
        output = model.sentiment(recom_prods)
        return render_template('index.html', tables=[output.to_html(classes='data',index=False)])
    else:
        return render_template('index.html', message=recom_prods)
        


if __name__ == "__main__":
    app.run(debug=True)