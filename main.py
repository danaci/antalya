
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import os

import webbrowser
from openpyxl import load_workbook

import streamlit as st
import streamlit.components.v1 as components

###
import streamlit as st
from PIL import Image

st.markdown(" #### Controlling Machine Learning Algorithms with a Visual Interface")
st.divider()
col11,col12=st.columns(2)
with col11:
    st.warning("Provincial Directorate of Agriculture - MERSİN :cityscape:\n http://mersin.tarim.gov.tr")
with col12:
    st.success("Halil DANACI-\nGeology Eng.:male-judge: https://github.com/danaci")

##
image = Image.open("regression.png") #Image name
fig = plt.figure()
plt.imshow(image)
st.pyplot(fig)

####
st.sidebar.caption("Options")
dosyalar = os.listdir(".")
liste = []
for dosya in dosyalar:
    if dosya.endswith(".xlsx") or dosya.endswith(".txt") or dosya.endswith(".csv") or dosya.endswith(".json"):
        liste.append(dosya)
dosya = st.sidebar.selectbox("File? ( xlsx, csv, txt) :file_folder:", sorted(liste))
if dosya.endswith(".xlsx"):#okunacak dosya türüne göre işlem
    data_set=pd.read_excel(dosya)
elif dosya.endswith(".csv") or dosya.endswith(".txt"):
    data_set=pd.read_csv(dosya)


col1, col2 = st.columns(2) # ekranı 2 kolona ayırıyoruz
with col1:
    st.success(f"File : {dosya} :open_file_folder:")
with col2:
    st.success(f"File Shape datas: {data_set.shape}:bookmark_tabs:")


if st.sidebar.checkbox("Data Edit Mode:pencil2:"):
    st.write("All Records ")
    st.data_editor(data_set)

else:
    st.write("All Records :large_yellow_square:=Max:large_red_square:=Min")
    st.dataframe(data_set.style.highlight_max(color="yellow",axis=0).highlight_min(color="red",axis=0)) #data setini ekrana gösteriyoruz
bag_deg = st.sidebar.selectbox("dependent field:pouch:", (data_set.columns))#colonlar listeye atılıyor
y=data_set[bag_deg]#bağımlı değişken seçtiriliyor
data_set.drop([bag_deg], axis=1, inplace=True)#seçilen kolon siliniyor
x = data_set#geri kalan x değişkenine aktarılıyor
# *************
test_size=st.sidebar.number_input("Test Size?:signal_strength:",min_value=.10,max_value=.95)
rnd_state=st.sidebar.slider("Random State Value:wastebasket:",max_value=100,min_value=0,value=42)
max_depth=st.sidebar.number_input("Max Depth?:arrow_up_small:",min_value=1,max_value=20)
alpha=st.sidebar.number_input("alpha?",min_value=1,max_value=20)

# *************
if st.sidebar.checkbox("Data Set Statistics:chart_with_downwards_trend:"):
    st.warning("Data Set Statistics")
    st.write(data_set.describe())#data seti istatistiklerini ekrana gösteriyoruz

if st.sidebar.checkbox("Missing Data Analysis:skin-tone-3:"):
    st.success("Missing Data Analysis")
    st.write(data_set.isnull().sum())#eksik veri analizi

if st.sidebar.checkbox("Distribution of Data:writing_hand:",):
    st.success("Distribution of Data")
    for i in data_set.columns:
        i=data_set[i].value_counts()
        st.dataframe(i)

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=test_size, random_state=rnd_state)

model_sec=st.sidebar.selectbox("Select Model :clipboard:",("Decision Tree Regressor",
                                                "Random Forest Regressor",
                                                "lasso Reg",
                                                "Elastic Regressor",
                                                "Ridge Regressor"))

if model_sec=="Decision Tree Regressor":
    st.write("Decision Tree Regressor column:", bag_deg)
    tree_regression = DecisionTreeRegressor(random_state=rnd_state, max_depth=max_depth)
    tree_regression = tree_regression.fit(x_train, y_train)
    tahmin_tree_regression = tree_regression.predict(x_test)
    st.write(tahmin_tree_regression)
    mae = mean_absolute_error(y_test, tahmin_tree_regression)
    mse = mean_squared_error(y_test, tahmin_tree_regression)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = r2_score(y_test, tahmin_tree_regression)
    st.dataframe({
        "Algorithm":"Decision Tree",
        "Mean Absolute Error":mae,
        "Mean Squared Error": mse,
        "Rmse": rmse,
        "r2": r2})

if model_sec=="Random Forest Regressor":
    st.write("Random Forest Regressor column:",bag_deg)
    random_regression = RandomForestRegressor(max_depth=max_depth, random_state=rnd_state)
    random_regression.fit(x_train, y_train)
    tahmin_random_regression = random_regression.predict(x_test)
    st.write(tahmin_random_regression)
    mae = mean_absolute_error(y_test, tahmin_random_regression)
    mse = mean_squared_error(y_test, tahmin_random_regression)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = r2_score(y_test, tahmin_random_regression)
    st.dataframe({
        "Algorithm": "Random Forest Regressor",
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Rmse": rmse,
        "r2": r2
    })
if model_sec=="lasso Reg":
    st.write("Lasso Regressor column:", bag_deg)
    lassoReg = Lasso(alpha=alpha)
    lassoReg.fit(x_train, y_train)
    tahmin_lasso = lassoReg.predict(x_test)
    st.write(tahmin_lasso)
    mae = mean_absolute_error(y_test, tahmin_lasso)
    mse = mean_squared_error(y_test, tahmin_lasso)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = r2_score(y_test, tahmin_lasso)
    st.dataframe({
        "Algorithm": "lasso Reg",
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Rmse": rmse,
        "r2": r2
    })
if model_sec=="Elastic Regressor":
    rnd_state=0
    st.write("Elastic Regressor column:", bag_deg)
    elastic_reg = ElasticNet(random_state=rnd_state)
    elastic_reg.fit(x_train, y_train)
    tahmin_elastic = elastic_reg.predict(x_test)
    st.write(tahmin_elastic)
    mae = mean_absolute_error(y_test, tahmin_elastic)
    mse = mean_squared_error(y_test, tahmin_elastic)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = r2_score(y_test, tahmin_elastic)
    st.dataframe({
        "Algorithm": "Elastic Regressor",
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Rmse": rmse,
        "r2": r2
    })
if model_sec=="Ridge Regressor":
    st.write("Ridge Regressor column:", bag_deg)
    ridge_reg = Ridge()
    ridge_reg.fit(x_train, y_train)
    tahmin_ridge = ridge_reg.predict(x_test)
    st.write(tahmin_ridge)
    mae = mean_absolute_error(y_test, tahmin_ridge)
    mse = mean_squared_error(y_test, tahmin_ridge)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = r2_score(y_test, tahmin_ridge)
    st.dataframe({
        "Algorithm": "Ridge Regressor",
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Rmse": rmse,
        "r2": r2
    })



##### chatgpt entegrasyonu
st.subheader('Asistant')
st.link_button("ChatGPT", "https://chat.openai.com",use_container_width=300,type="primary")


#####

#dosya export işlemleri
st.success("Export file:newspaper:")
st.divider()
kolonlar=st.multiselect("select export columns",data_set.columns,placeholder="select columns")
df2=pd.DataFrame(data_set,columns=kolonlar)
csv_dosya=st.text_input("Create csv file name?")
if st.button("save csv :pencil2:"):
    df2.to_csv(csv_dosya+".csv",index=False)

if st.sidebar.button(f" Delete {dosya}? "):#işe yaramayan dosyaları siliyoruz
    os.remove(dosya)
    st.rerun()

