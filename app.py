import pandas as pd 
import numpy as np 

import streamlit as st 
import pickle as pk 
model=pk.load(open('car_sales_random_forest.pkl','rb'))



def car_price(Present_Price, Kms_Driven, Owner, New_year, Fuel_Type_Diesel,Fuel_Type_Petrol, Seller_Type, Transmission_Manual):
       input=np.array([[Present_Price, Kms_Driven, Owner, New_year, Fuel_Type_Diesel,Fuel_Type_Petrol, Seller_Type, Transmission_Manual]])
       
       prediction=model.predict(input)
       return prediction


def main():

    st.title("** CAR SALES PREDICTION **")
    html="""<div style= "background-color:red" ;padding: 15px"">
        <h2><b> CAR SALES MACHINE LEARNING MODEL<b> </h2>
    </div>
    """
    st.markdown(html,unsafe_allow_html=True)
    Present_Price=st.text_input('Present Car Price  ')
    Kms_Driven=st.text_input('Kilometer Driven by The car  ')
    Owner=st.selectbox('How Many owners Previously had the car, (0 or 1 or 3)',[0,1,3])
    New_year=st.text_input('Years of the car')
    Fuel_Type_Diesel=st.selectbox('Diesel fuel type ,',['Yes','No'])
    if (Fuel_Type_Diesel=='Yes'):
        Fuel_Type_Diesel=1
    else:
        Fuel_Type_Diesel=0
    Fuel_Type_Petrol=st.selectbox('Petrol fuel type ',['YES','NO'])
    if Fuel_Type_Petrol=='YES':
        Fuel_Type_Petrol=1
    else:
        Fuel_Type_Petrol=0
    Seller_Type=st.selectbox('Seller Type ',['INDIVIDUAL','DEALER'])
    if Seller_Type=='INDIVIDUAL':
        Seller_Type=1
    else:
        Seller_Type=0
    Transmission_Manual=st.selectbox('Trasmission_Manual ',['MANUAL','AUTOMATIC'])
    if Transmission_Manual=='MANUAL':
        Transmission_Manual=1
    else:
        Transmission_Manual=0
    st.write('*')


    if st.button('Predict the car price'):
        data_predict=car_price(Present_Price, Kms_Driven, Owner, New_year, Fuel_Type_Diesel,Fuel_Type_Petrol, Seller_Type, Transmission_Manual)
        st.success('The current Price of the car is {}'.format(data_predict[0]))

if __name__=='__main__':
    main()


st.write('Machine Learning Model Developed By Emmanuel Oladejo')
st.title('DISCLAIMER !!!')
st.write('This is an Artificial Model Prediction and is not tottaly Accurate')


st.write('Link to the Machine Learning Models')
st.write('https://github.com/EMMANUEL1111/Car-Sales/blob/main/CAR%20SALES.ipynb')