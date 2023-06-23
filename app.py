import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack

app = Flask(__name__)

with open('models/xgb_model.pkl', 'rb') as f:
    vectorizer1, vectorizer2, vectorizer3, vectorizer4, t_vectorizer1, t_vectorizer2, xgb_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    X_test_brand_OH = vectorizer1.transform([request.form.get('brand_name')])
    X_test_cat1_OH = vectorizer2.transform([request.form.get('cat_1')])
    X_test_cat2_OH = vectorizer3.transform([request.form.get('cat_2')])
    X_test_cat3_OH = vectorizer4.transform([request.form.get('cat_3')])

    X_test_name_tfidf = t_vectorizer1.transform([request.form.get('name')])
    X_test_description_tfidf = t_vectorizer2.transform([request.form.get('item_description')])

    X_test_cat = csr_matrix(pd.get_dummies(pd.DataFrame({'shipping': [request.form.get('shipping')], 'item_condition_id': [request.form.get('item_condition')]}), sparse=True).values)

    X_test_final = hstack((X_test_brand_OH, X_test_cat1_OH, X_test_cat2_OH, X_test_cat3_OH, X_test_name_tfidf, X_test_description_tfidf, X_test_cat)).tocsr()

    prediction = xgb_model.predict(X_test_final) 
    result = prediction[0]

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)