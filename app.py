from flask import Flask, request, render_template, url_for
import numpy as np
import pickle

# Load the pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))  # Load saved scaler

# Creating Flask app
app = Flask(__name__, static_folder= 'static')

@app.route('/')
def index():
    return render_template("WebPage.html")

@app.route('/about')
def about():
    return render_template('about-page.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosphorus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Prepare data for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Apply MinMaxScaler and StandardScaler
    scaled_features = ms.transform(single_pred)  # Use transform, not fit
    prediction = model.predict(scaled_features)

    # Define crop dictionary
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Get the prediction result
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{}".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Pass the result and input values back to the template
    return render_template(
        'WebPage.html',
        result=result,
        Nitrogen=N,
        Phosphorus=P,
        Potassium=K,
        Temperature=temp,
        Humidity=humidity,
        Ph=ph,
        Rainfall=rainfall
    )

# Python main
if __name__ == "__main__":
    app.run(debug=True)
