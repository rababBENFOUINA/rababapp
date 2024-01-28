from flask import  render_template, request
from application import app
import pandas as pd
from joblib import load

# Chargez le modèle au démarrage de l'application
model = load("application\\best_model.joblib")

@app.route("/book_recommendation_form", methods=['POST'])
def book_recommendation_form():
    publish_year = request.form.get('publish_year')
    ratings_count = request.form.get('ratings_count')
    pages_no = request.form.get('pages_no')
    
    category_encoded = request.form.get('category_encoded')

    new_user_features = pd.DataFrame({ 'year':[2000], 'pages':[200], 'category_encoded':[2000]  ,'avg_rating':[1]  })


    new_user_features = pd.DataFrame({'year': [publish_year], 'pages': [pages_no], 'category_encoded': [pages_no],
                                      'avg_rating': [ratings_count]})
    
    probability = model.predict_proba(new_user_features)[0, 1]  
    percentage_recommendation = probability * 100
    formatted_percentage = round(percentage_recommendation, 2)

    recommendation = model.predict(new_user_features)

    recommendation = model.predict(new_user_features)
    return render_template('result.html', recommendation=recommendation[0] , percentage_recommendation=formatted_percentage)
