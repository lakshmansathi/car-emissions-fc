import streamlit as st
import pandas as pd
import numpy
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
st.set_option('deprecation.showPyplotGlobalUse', False) 
co2 = RandomForestRegressor(max_depth = 20, n_jobs = 1)
nox = ElasticNet(alpha = 0.05,max_iter = 50)
fuel_cost = RandomForestRegressor(max_depth = 20, n_jobs = 1)
co_emissions = RandomForestRegressor(max_depth = 20, n_jobs = 1)
thc = RandomForestRegressor(max_depth = 20, n_jobs = 1)


data = pd.read_csv('dataset.csv')
data = data.iloc[: , 1:]
target = pd.read_csv('co2_target.csv')
target_thc = pd.read_csv('thc_target.csv')
target_thc = target_thc.iloc[: , 1:]
manufacturer_df = pd.read_csv('manuf_labelled.csv')
transmission_df = pd.read_csv('transmission_labelled.csv')
manufacturer_df = manufacturer_df.iloc[: , 1:]
transmission_df = transmission_df.iloc[: , 1:]
target_nox = pd.read_csv('nox_target.csv')
target_nox = target_nox.iloc[: , 1:]
target_fc = pd.read_csv('fc_target.csv')
target_fc = target_fc.iloc[: , 1:]
target = target.iloc[: , 1:]
target_co = pd.read_csv('co_target.csv')
target_co = target_co.iloc[: , 1:]
st.header('Emissions Car APP')
st.write(data)
st.write('---')
X = data[['manufacturer','transmission','engine_capacity','noise_level','euro_standard_','Petrol','Diesel','Petrol Hybrid']]
X["engine_capacity"] = pd.to_numeric(X["engine_capacity"], downcast="float")
X["noise_level"] = pd.to_numeric(X["noise_level"], downcast="float")
X["euro_standard_"] = pd.to_numeric(X["euro_standard_"], downcast="float")
X["manufacturer"] = pd.to_numeric(X["manufacturer"], downcast="float")
X["transmission"] = pd.to_numeric(X["transmission"], downcast="float")
X["Petrol"] = pd.to_numeric(X["Petrol"], downcast="float")
X["Diesel"] = pd.to_numeric(X["Diesel"], downcast="float")
X["Petrol Hybrid"] = pd.to_numeric(X["Petrol Hybrid"], downcast="float")
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
def user_input_features():
    Engine_Capacity = float(st.sidebar.slider('engine Capacity', float(X.engine_capacity.min()), float(X.engine_capacity.max()), float(X.engine_capacity.mean())))
    Noise_level = float(st.sidebar.slider('Noise_level', float(X.noise_level.min()), float(X.noise_level.max()), float(X.noise_level.mean()))
    euro_standard_ = float(st.sidebar.slider('euro_standard_', float(X.euro_standard_.min()), float(X.euro_standard_.max()), float(X.euro_standard_.mean()))
    petrol = st.sidebar.selectbox('Petrol', ('Yes','No'))
    Diesel = st.sidebar.selectbox('Diesel',  ('Yes','No'))
    Petrol_Hybrid = st.sidebar.selectbox('Petrol Hybrid',  ('Yes','No'))
    transmission = st.sidebar.selectbox('transmission', ('m5', 'sat5', 'm6', 'a4', 'a5', 'qm6', 'qa5', 'qm5', 'qm', 'qa', 'fa5', 'fm5', 'fa4', 'a3', 'cvt', 'a', '4ss', 'm4', 'sat_5', '5mt', '4at', 'm5x2', 'm5/s', 'sm6', 'a6', 'mt', 'at', 'm5n', 'a4l', 'm5ne', 'a4ne', '3at', '5mtx2', '4atx2', 'a4x2', 'm', 'av', 'qa6', 'ss5', 's/a6', 'm3', 'mta', 'sat6', 'hybrid', 'qd6', 'd6', 'a/sat5', 'asm', 'a5x2', 'm6/s6', 'a7', 'multi5', 'e-cvt', 'electric', 'smg_7', '6mt', '5a/tx2', 'm6x2', '5at', '6amt', '5_amt', 'mta5', 'i-shift', 'a8', 'dct7', 'm6-awd', 's6', 'a6x2', 'd7', 'm7', 'a5-awd', 'multidrive', 'mta6', 'qd7', 'a6-awd', 'dm6', 'mcvt', 'semi-auto', 'am5', 'multi6', 'multidriv', 'dct6', 'et5', 'qa8', '6at', 'amt6', 'amt5', '7sp._ssg', 'asm__', 'nan', 'm8', 'a8-awd', 'qd8', 'a2/6', 'a9', 'mps6', 'mps6awd', '8at', 'mps6-awd', 'a7-awd', 'a9-awd', 'qa7', '10at', 'sa6', '7dct', '6-speed_auto_dct', 'mt5', 'mt6', '10_speed_automatic', '8a-awd', '8a_awd', 'dct8', 'manual'))
    manufacturer = st.sidebar.selectbox('manufacturer',  ('abarth', 'alfa_romeo', 'aston_martin_lagonda', 'audi', 'bentley_motors', 'bmw', 'chrysler_jeep', 'citroen', 'ds', 'ferrari', 'fiat', 'ford', 'honda', 'hyundai', 'jaguar', 'kia', 'lamborghini', 'lexus', 'mazda', 'mclaren', 'mercedes-benz', 'mini', 'mitsubishi', 'nissan', 'peugeot', 'porsche', 'renault', 'rolls_royce', 'seat', 'skoda', 'smart', 'ssangyong', 'subaru', 'suzuki', 'toyota', 'vauxhall', 'volkswagen', 'volvo', 'dacia', 'infiniti', 'land_rover', 'london_taxi_company', 'maserati', 'tesla', 'volkswagen_c.v.', 'lotus', 'morgan_motor_company', 'chevrolet', 'cadillac', 'mg_motors_uk'))

    data = {
            'transmission': transmission,
            'manufacturer':manufacturer,
            'Engine_Capacity': Engine_Capacity,
            'Noise_level': Noise_level,
            'Euro_Standard': euro_standard_,
            'petrol': petrol,
            'Diesel': Diesel,
            'Petrol Hybrid' : Petrol_Hybrid
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()
st.header('Specified Input parameters')
st.write(df)
st.write('---')
manufacturer_dict_final = {}
def manufacturer_dict(manufacturer_df):
    manufacturer_dict =manufacturer_df.set_index('key').transpose().to_dict(orient='records')
    for value in manufacturer_dict:
        manufacturer_dict_final.update(value)
    return manufacturer_dict_final
transmission_dict_final = {}
def transmission_dict(transmission_df):
    transmission_dict =transmission_df.set_index('key').transpose().to_dict(orient='records')
    for value in transmission_dict:
        transmission_dict_final.update(value)
    return transmission_dict_final
target_mapper_manuf = manufacturer_dict(manufacturer_df)
def target_encode_manuf(val):
    return target_mapper_manuf[val]

target_mapper_transmission = transmission_dict(transmission_df)
def target_encode_transmission(val):
    return target_mapper_transmission[val]
dict
df['transmission'] = df['transmission'].apply(target_encode_transmission)
df['manufacturer'] = df['manufacturer'].apply(target_encode_manuf)





y = target['co2']
co2.fit(X, y)
X_train,X_test,y_train,y_test=train_test_split(X,y, random_state = 0,test_size=0.2)
y_nox= target_nox[['emissions_nox_[mg/km]']]
nox.fit(X, y_nox)
X_train_nox,X_test_nox,y_train_nox,y_test_nox=train_test_split(X,y_nox, random_state = 0,test_size=0.2)
y_fc= target_fc[['fuel_cost_10000-12000_miles']]
fuel_cost.fit(X, y_fc)
X_train_fc,X_test_fc,y_train_fc,y_test_fc=train_test_split(X,y_fc, random_state = 0,test_size=0.2)
y_co= target_co[['emissions_co_[mg/km]']]
co_emissions.fit(X, y_co)
X_train_co,X_test_co,y_train_co,y_test_co=train_test_split(X,y_co, random_state = 0,test_size=0.2)
y_thc= target_thc[['thc_emissions_[mg/km]']]
thc.fit(X, y_thc)
X_train_thc,X_test_thc,y_train_thc,y_test_thc =train_test_split(X,y_thc, random_state = 0,test_size=0.2)

dict={'Yes':1.0,'No':0.0}
df['petrol']=df['petrol'].map(dict)
df['Diesel']=df['Diesel'].map(dict)
df['Petrol Hybrid']=df['Petrol Hybrid'].map(dict)
co2_result = co2.score(X_test, y_test)
nox_result = nox.score(X_test_nox, y_test_nox)
fuel_cost_result = fuel_cost.score(X_test_fc,y_test_fc)
thc_result = thc.score(X_test_thc,y_test_thc)
co_emissions_result = co_emissions.score(X_test_co,y_test_co)
st.header('Score Predictions')
st.write('CO2 score')
st.write(co2_result)
st.write('NOX score')
st.write(nox_result)
st.write('CO score')
st.write(co_emissions_result)
st.write('THC score')
st.write(thc_result)
st.write('Fuel Cost score')
st.write(fuel_cost_result)
st.write('---')

co2_predict = co2.predict(df)
nox_predict = nox.predict(df)
fuel_cost_predict = fuel_cost.predict(df)
co_emissions_predict = co_emissions.predict(df)
thc_predict = thc.predict(df)

st.header('Outcome Predictions')
st.write('CO2 prediction')
st.write(co2_predict)
st.write('NOX prediction')
st.write(nox_predict)
st.write('CO prediction')
st.write(co_emissions_predict)
st.write('THC prediction')
st.write(thc_predict)
st.write('Fuel Cost prediction')
st.write(fuel_cost_predict)
st.write('---')
# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap

#model = RandomForestRegressor(max_features = 'sqrt',max_depth = 7, n_estimators = 58, n_jobs = 1)
#model.fit(X, y)

#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(X)

#st.header('Feature Importance')
#shap.summary_plot(shap_values, X)
#st.pyplot(bbox_inches='tight')
#st.write('---')

#plt.title('Feature importance based on SHAP values (Bar)')
#shap.summary_plot(shap_values, X, plot_type="bar")
#st.pyplot(bbox_inches='tight')


