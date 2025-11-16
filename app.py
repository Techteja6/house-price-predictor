from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    size = float(request.form['size'])
    bedrooms = int(request.form['bedrooms'])
    prediction = model.predict([[size, bedrooms]])
    return render_template('index.html',
                           result=f"🏡 Estimated Price: ₹{prediction[0]:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
