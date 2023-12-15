#Importing Libraries
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
import streamlit as st
symptoms=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
        'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
        'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
        'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
        'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
        'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
        'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
        'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
        'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
        'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
        'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
        'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
        'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
        'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
        'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
        'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
        'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
        'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
        'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
        'yellow_crust_ooze']
with st.form("my_form"):
    s1=st.selectbox("Enter symptom:",symptoms, key="s1",index=None,)

    s2=st.selectbox("Enter symptom:",symptoms, key="s2",index=None)

    s3=st.selectbox("Enter symptom:",symptoms, key="s3",index=None)

    s4=st.selectbox("Enter symptom:",symptoms, key="s4",index=None)

    s5=st.selectbox("Enter symptom:",symptoms, key="s5",index=None)
    submitted = st.form_submit_button("Submit")
    if s1 and s2 and submitted:
       #List of the symptoms is listed here in list l1.
            
        l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
            'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
            'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
            'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
            'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
            'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
            'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
            'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
            'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
            'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
            'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
            'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
            'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
            'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
            'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
            'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
            'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
            'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
            'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
            'yellow_crust_ooze']

        disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
            'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
            'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
            'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
            'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
            'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
            'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
            'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
            'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
            'Osteoarthristis', 'Arthritis',
            '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
            'Urinary tract infection', 'Psoriasis', 'Impetigo']

        l2=[]
        for i in range(0,len(l1)):
            l2.append(0)
        print(l2)
        #Reading the  testing.csv file
        tr=pd.read_csv("testingset.csv")
        df=pd.read_csv("trainingset.csv")

        df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
            'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
            'Migraine':11,'Cervical spondylosis':12,
            'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
            'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
            'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
            'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
            '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
            'Impetigo':40}},inplace=True)
        X= df[l1]
        y = df[["prognosis"]]
        np.ravel(y)

        #Using inbuilt function replace in pandas for replacing the values

        tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
            'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
            'Migraine':11,'Cervical spondylosis':12,
            'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
            'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
            'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
            'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
            '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
            'Impetigo':40}},inplace=True)
        tr.head()
        X_test= tr[l1]
        y_test = tr[["prognosis"]]
        np.ravel(y_test)
        # print(X_test)
        with open("fin.pkl", "rb") as file:
            model = pickle.load(file)
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import VotingClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix
        gnb = GaussianNB()
        # pred1="ab"
        gnb=gnb.fit(X,np.ravel(y))
                # y_pred_gnb = gnb.predict(X_test)

                # k-Nearest Neighbors
        knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
        knn=knn.fit(X,np.ravel(y))
                # y_pred_knn = knn.predict(X_test)

                # Create an ensemble model using majority voting
                # ensemble_model = VotingClassifier(estimators=[('gnb', gnb), ('knn', knn)], voting='hard')
                # # Fit the ensemble model on the training data
                # ensemble_model.fit(X, np.ravel(y))

                # # Make predictions on the test data
        y_pred_ensemble = model.predict(X_test)

                #Save model
                # model_filename = 'ensemble_model.pkl'
                # joblib.dump(ensemble_model, model_filename)



                # model_filename = 'ensemble_model.pkl'
                # joblib.dump(ensemble_model, model_filename)


                # saveData()
                # import pickle
                # model_filename = 'ensemble_model.pkl'
                # with open(model_filename, 'wb') as file:
                #     pickle.dump(ensemble_model, file)
                # name='ensemble.pkl'
                # pickle.dump(ensemble_model,open(name,'wb'))
                # from sklearn.externals import joblib
                # model_filename = 'ensemble_model.pkl'
                # joblib.dump(ensemble_model, model_filename)
                
                # Print accuracy and confusion matrix for the ensemble model
        # print("Ensemble Model (GNB + KNN)")
        # print("Accuracy")
        # print(accuracy_score(y_test, y_pred_ensemble))
        # print("Confusion matrix")
        conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)
        print(conf_matrix_ensemble)


                # Collect symptoms directly from user input
        # psymptoms = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever']
        psymptoms=[s1,s2,s3,s4,s5]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1
        inputtest = [l2]

                # Make predictions on the test data
        y_pred_ensemble = model.predict(inputtest)
                
                # Print accuracy and confusion matrix for the ensemble model
        # print("Ensemble Model (GNB + KNN)")
        # print("Predicted Disease Index:", y_pred_ensemble[0])
        # print("Disease List:", disease)

        predicted = y_pred_ensemble[0]
        if 0 <= predicted < len(disease):
            print(disease[predicted])
        else:
            print("Not Found")
        # st.write(str(s1),s2,s3,s4,s5)
        st.write("the suspected disease is: ")
        st.header(disease[predicted])
    elif submitted:
        st.error('Fill atleast 2 symptoms', icon="🚨")





            
        


