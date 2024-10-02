import numpy
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


def classification(data):
                    
    algorithme=st.sidebar.radio("Choisir un algorithme : ",("LogisticRegression","KNeighborsClassifier","Naive Bayes","DecisionTreeClassifier","RandomForestClassifier","AdaBoostClassifier"))
    prediction=st.sidebar.toggle('Commencer la prediction')

    array = data.values

    X = array[ : , 0:-1]
    Y = array[ : , -1]
    test_proportion = 0.3
    seed = 8

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_proportion, random_state = seed)


    models = []
    models.append(('LogisticRegression', LogisticRegression(solver='newton-cg')))
    models.append(('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=1)))
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('RandomForestClassifier', RandomForestClassifier()))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))

    for model_name, model in models:
        if algorithme == model_name:
            model.fit(X_train, Y_train)

            st.write(f'Confusion Matrix\n------------------')
            predict = model.predict(X_test)
            matrix = confusion_matrix(Y_test, predict)
            st.write(matrix)

            st.write(f'Evaluation Metrics\n-------------------')
            acc, prec, rec, f1, a_r = metrics(matrix)
            st.write(f'Accuracy: {(acc*100).round(2)} %')
            st.write(f'Precision: {(prec*100).round(2)} %')
            st.write(f'Recall: {(rec*100).round(2)} %')
            st.write(f'F1-Score: {(f1*100).round(2)} %')
            st.write(f'AUC-ROC: {(a_r*100)} %')

    if prediction:
        st.sidebar.write("Entrer les valuers pour la prediction")
        values=[]
        for i in range(len(data.columns)-1):
            val = st.sidebar.number_input(f"Entrez la valuer {i+1}")
            values.append(val)

        if st.sidebar.button('Done'):   
            new_data = numpy.array([values])
            for model_name, model in models:
                if algorithme == model_name:
                    model.fit(X_train, Y_train)   
                    prediction2 = model.predict(new_data)
                    proba = model.predict_proba(new_data)
                    st.write("# Prediction")
                    st.write(int(prediction2[0]), proba)

    if st.sidebar.button('Save Model'):  
        for model_name, model in models:
                if algorithme == model_name:
                    model.fit(X_train, Y_train)   
                    st.write(f'### Saving the model.......')
                    model_name = 'Classification.pickle'
                    pickle.dump(model, open(model_name, 'wb'))
                    st.write(f'###### Model saved!')


                    

def metrics(matrix):
    TN, FN, FP, TP = matrix.ravel()
    acc = (TN + TP) / (TP + TN + FP + FN)
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    spec = TN / (TN + FP)
    a_r = 0 if acc*100 < 50 else 1
    f1 = 2 * ((prec * rec) / (prec + rec))
    return acc, prec, rec, f1, a_r
