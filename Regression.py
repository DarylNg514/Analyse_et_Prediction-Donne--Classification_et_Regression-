import streamlit as st
import numpy
import pickle
# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def regression(data):
                    
    algorithme=st.sidebar.radio("Choisir un algorithme : ",("LinearRegression","Ridge","Lasso","RandomForestRegressor","GradientBoostingRegressor","DecisionTreeRegressor","SVR"))
    prediction=st.sidebar.toggle('Commencer la prediction')
    array = data.values

    X = array[ : , 0:-1] 
    Y = array[ : , -1]
    test_proportion = 0.3
    seed = 8

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_proportion, random_state = seed)

    metrics = []
    metrics.append(('mean absolute error', mean_absolute_error))
    metrics.append(('mean squared error', mean_squared_error))
    metrics.append(('r2 score', r2_score))

    models = []
    models.append(('LinearRegression', LinearRegression()))
    models.append(('Ridge', Ridge()))
    models.append(('Lasso', Lasso()))
    models.append(('RandomForestRegressor', RandomForestRegressor()))
    models.append(('GradientBoostingRegressor', GradientBoostingRegressor()))
    models.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
    models.append(('SVR', SVR()))
    
    st.write(f'Model Evaluation\n----------------')
    for metric_name, metric in metrics:
        for model_name, model in models:
            if algorithme == model_name:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)
                error = metric(Y_test, y_pred)
                st.write(f'{metric_name} : {error.round(2)}')
    if prediction:
        st.sidebar.write("Entrer les valuers pour la prediction")
        values=[]
        for i in range(len(data.columns)-1):
            val = st.sidebar.number_input(f"Entrez la valuer {i+1}")
            values.append(val)

        if st.sidebar.button('Done'):   
            for model_name, model in models:
                if algorithme == model_name:
                    new_data = numpy.array([values])
                    model.fit(X_train, Y_train)
                    y_pred = model.predict(X_test)
                    MAE = mean_absolute_error(Y_test, y_pred)
                    st.write("# Prediction")
                    new_output = model.predict(new_data)
                    st.write(f'Predicted value: {new_output}')
                    st.write(f'Predicted value intervale: [{(new_output-MAE).round(2)}, {new_output+MAE}]')


    if st.sidebar.button('Save Model'):  
        for model_name, model in models:
                if algorithme == model_name:
                    model.fit(X_train, Y_train)   
                    st.write(f'### Saving the model')
                    model_name = 'Regression.pickle'
                    pickle.dump(model, open(model_name, 'wb'))
                    st.write(f'###### Model saved!')
                                            
    
    

