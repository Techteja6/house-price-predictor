import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample data
data = pd.DataFrame({
    'size': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 5],
    'price': [200000, 300000, 400000, 500000, 600000]
})

X = data[['size', 'bedrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model/model.pkl', 'wb'))
print("✅ Model trained & saved as model.pkl")
