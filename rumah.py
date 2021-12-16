import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import time
import requests
import json
from managedb import *
import hashlib
from id3 import Id3Estimator
from id3 import export_graphviz
import subprocess
from PIL import Image


Error = """
<style>
.stException {
	visibility:hidden;
}
</style>



"""


st.markdown(Error, unsafe_allow_html=True)
def generate_hash(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hash(password,hashed_text):
	if generate_hash(password) == hashed_text :
		return hashed_text
	return False


def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()

@st.cache
def load_data():
	df=pd.read_excel('data.xls')
	df=df.drop(['country'],axis=1)
	df=df[df['harga']>0]
	df.rename(columns={'statezip':'zip'}, inplace=True)
	df['zip']=df['zip'].str.replace('WA','').astype(int)
	df['lantai']=df['lantai'].astype(int)
	df=df[df['kamar']>0]
	df=df[df['kamar_mandi']>0]
	return df

df=load_data()

@st.cache
def get_locations(zip):
	for i in zip:
		url='https://data.opendatasoft.com/api/records/1.0/search/?dataset=geonames-postal-code%40public&q=&rows=10&facet=postal_code&refine.country_code=US&refine.postal_code='+str(i).format(zip)
		data=requests.get(url).json()
		lat=data['records'][0]['fields']['latitude']
		lng=data['records'][0]['fields']['longitude']
	return lat, lng

		

test_size = 0.25



def main():
	submenu = ["Home","SPK","Add User","Lihat User"]
	username = st.sidebar.text_input("Masukan Username")
	password = st.sidebar.text_input("Masukan Password",type='password')
	if st.sidebar.checkbox("Login"):
		create_usertable()
		hash_pasword = generate_hash(password)
		result = login_user(username,verify_hash(password,hash_pasword))
		if result:
			st.success("Welcome {}".format(username))
				
			if username == "admin":
				pilih = st.sidebar.selectbox("Menu",submenu)
				if pilih == "Home":
					st.subheader('Kelompok 2')
					st.subheader('Pencarian Harga Rumah Berdasarkan Kriteria')
					st.write("Ini adalah web untuk melakukan analisa terhadap harga rumah berdasarkan kriteria yang anda inginkan.")
					st.write("berikut adalah dataset yang sudah kami kumpulkan untuk melakukan analisa rumah menggunakan metode Decision tree")
					df=load_data()
					df
				elif pilih == "SPK":
					df=load_data()
					y1=df['harga']
					X1=df[['kamar','kamar_mandi','lantai','luas_rumah','kondisi']]
					st.subheader('SPK PENCARIAN APARTEMENT')
					st.write("Jika Tombol Cari Tidak Ada Maka Data yang anda cari belum ada pada database Kami")

					params={
					'kamar' : st.selectbox('Kamar.',(1,2,3)),
					'kamar_mandi' : st.selectbox('Kamar Mandi.',(1,2,3)),
					'lantai' : st.selectbox('Jumlah Lantai.',(1,2)),
					'sqft' : st.selectbox('Perkiraan Luas Tanah m²', (1000,1500,2000,2500,3000,4000,5000)),
					'kondisi' : st.selectbox('Kondisi.',('Sangat Bagus','Bagus','Baik','Lumayan','Jelek')),
					}

					if params['kondisi']=='Sangat Bagus':
						params['kondisi']=5
					elif params['kondisi']=='Bagus':
						params['kondisi']=4
					elif params['kondisi']=='Baik':
						params['kondisi']=3
					elif params['kondisi']=='Lumayan':
						params['kondisi']=2
					elif params['kondisi']=='Jelek':
						params['kondisi']=1
					else:
						pass
					df=df[df['kamar']==params['kamar']]
					df=df[df['kamar_mandi']==params['kamar_mandi']]
					df=df[df['lantai']==params['lantai']]
					df=df[df['kondisi']==params['kondisi']]
					df=df[(df['luas_rumah']>0.01*params['sqft']) & (df['luas_rumah']<1.01*params['sqft'])]
					df.reset_index()
					df['lat']=[get_locations(df.iloc[[i]]['zip'].values.astype(int))[0] for i in range(len(df))]
					df['lon']=[get_locations(df.iloc[[i]]['zip'].values.astype(int))[1] for i in range(len(df))]
					y=df['harga']
					X=df[['kamar','kamar_mandi','lantai','luas_rumah','kondisi']]
					feature_names = X.columns
					X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
					clf = Id3Estimator()
					clf.fit(X1, y1, check_input=True)
					dot_data = export_graphviz(clf.tree_,"dam.dot",feature_names)
					subprocess.call('dot -T png dam.dot -o out.png')
					models = DecisionTreeRegressor(max_depth=25)
					df_models = pd.DataFrame()
					temp = {}
					print(models)
					m = str(models)
					temp['Model'] = m[:m.index('(')]
					models.fit(X_train, y_train)
					MAE = sqrt(mse(y_test, models.predict(X_test)))
					PRED =models.predict(pd.DataFrame(params,  index=[0]))[0]
					df_models = df_models.append(df)
					data = list(range(1,11))
					if df_models.shape[0] > len(data):
						nomer = pd.DataFrame(data,columns=['peringkat'])
					else:
						data = list(range(1,df_models.shape[0]+1))
						nomer = pd.DataFrame(data,columns=['peringkat'])
					df_models = df_models.sort_values(by=['luas_rumah'])
					df_models = df_models.tail(10)
					bung = pd.concat([nomer.tail(10),df_models.reset_index()],axis=1)
					asli = bung
					asli = asli.drop(['index'],axis=1)
					asli['harga'] = asli['harga'].apply(np.int64)
					asli['kamar_mandi'] = asli['kamar_mandi'].apply(np.int64)
					asli = asli[['harga','kamar','kamar_mandi','luas_rumah','lantai','kondisi','Ket','alamat','kota']]
					
					

					
					btn = st.button("Cari")
					if btn:
						image = Image.open('out.png')
						st.image(image, caption='Diagram Decision Tree')
						st.write('Prediksi Harga Rumah Rata-Rata **${:.2f}**'.format(PRED))
						asli
						st.map(df_models[['lat','lon']])
						st.write('Berdasarkan hasil Pencarian, Telah ditemukan')
						for i in range(len(asli)):
							kota = asli.iloc[i]['kota']
							alamat = asli.iloc[i]['alamat']
							harga = asli.iloc[i]['harga']
							peringkat = bung.iloc[i]['peringkat']
							st.write(' ~ Peringkat Nomer **{}** Ditemukan Dikota **{}** Dengan Alamat **{}** dan dengan Harga **${:.2f}**'.format(peringkat,kota,alamat,harga))
						
					else:
						pass

					st.subheader('Informasi Tambahan')
					if st.checkbox('Show ML MAE'):
						st.write('MAE Harga = **{:.2f}**'.format(MAE))
					if st.sidebar.button('Show JSON'):
						st.json(df[['harga','kamar','kamar_mandi','lantai','luas_rumah','kondisi']].to_json())
					if st.sidebar.button('Close JSON'):
						asli
						for i in range(1,len(asli)):
							kota = asli.iloc[i]['kota']
							alamat = asli.iloc[i]['alamat']
							harga = asli.iloc[i]['harga']
							peringkat = bung.iloc[i]['peringkat']
							st.write('Berdasarkan hasil Pencarian, Peringkat Nomer {} Ditemukan Dikota {} Dengan Alamat {} dan dengan Harga {}'.format(peringkat,kota,alamat,harga))
				elif pilih == "Add User":
					new_username = st.text_input("User name")
					new_password = st.text_input("Password", type="password")
					confirm_password = st.text_input("Masukan Password Lagi", type='password')
					if new_password == confirm_password:
						st.success("Password cocok")
					else:
						st.warning("Password tidak cocok")
					if st.button("Submit"):
						create_usertable()
						hashed_new_password = generate_hash(new_password)
						add_userdata(new_username,hashed_new_password)
						st.success("Sucses Membuat akun")
						st.info("Silahkan Login Untuk Mencoba")
				elif pilih == "Lihat User":
					user_result = view_all_users()
					clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
					st.dataframe(clean_db)
			else:
				df=load_data()
				st.subheader('SPK PENCARIAN APARTEMEN')
				st.write("Jika Tombol Cari Tidak Ada Maka Data yang anda cari belum ada pada database Kami")

				params={
				'kamar' : st.selectbox('Kamar.',(1,2,3)),
				'kamar_mandi' : st.selectbox('Kamar Mandi.',(1,2,3)),
				'lantai' : st.selectbox('Jumlah Lantai.',(1,2)),
				'sqft' : st.selectbox('Perkiraan Luas Tanah m²', (1000,1500,2000,2500,3000,4000,5000)),
				'kondisi' : st.selectbox('Kondisi.',('Sangat Bagus','Bagus','Baik','Lumayan','Jelek')),
				}

				if params['kondisi']=='Sangat Bagus':
					params['kondisi']=5
				elif params['kondisi']=='Bagus':
					params['kondisi']=4
				elif params['kondisi']=='Baik':
					params['kondisi']=3
				elif params['kondisi']=='Lumayan':
					params['kondisi']=2
				elif params['kondisi']=='Jelek':
					params['kondisi']=1
				else:
					pass
				df=df[df['kamar']==params['kamar']]
				df=df[df['kamar_mandi']==params['kamar_mandi']]
				df=df[df['lantai']==params['lantai']]
				df=df[df['kondisi']==params['kondisi']]
				df=df[(df['luas_rumah']>0.01*params['sqft']) & (df['luas_rumah']<1.01*params['sqft'])]
				df.reset_index()
				df['lat']=[get_locations(df.iloc[[i]]['zip'].values.astype(int))[0] for i in range(len(df))]
				df['lon']=[get_locations(df.iloc[[i]]['zip'].values.astype(int))[1] for i in range(len(df))]
				y=df['harga']
				X=df[['kamar','kamar_mandi','lantai','luas_rumah','kondisi']]
					
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
				models = DecisionTreeRegressor(max_depth=25)
				df_models = pd.DataFrame()
				temp = {}
				print(models)
				m = str(models)
				temp['Model'] = m[:m.index('(')]
				models.fit(X_train, y_train)
				MAE = sqrt(mse(y_test, models.predict(X_test)))
				PRED =models.predict(pd.DataFrame(params,  index=[0]))[0]
				df_models = df_models.append(df)
				data = list(range(1,11))
				if df_models.shape[0] > len(data):
					nomer = pd.DataFrame(data,columns=['peringkat'])
				else:
					data = list(range(1,df_models.shape[0]+1))
					nomer = pd.DataFrame(data,columns=['peringkat'])
				df_models = df_models.sort_values(by=['harga'])
				df_models = df_models.tail(10)
				bung = pd.concat([nomer.tail(10),df_models.reset_index()],axis=1)
				asli = bung
				asli = asli.drop(['index'],axis=1)
				asli['harga'] = asli['harga'].apply(np.int64)
				asli['kamar_mandi'] = asli['kamar_mandi'].apply(np.int64)
				asli = asli[['harga','kamar','kamar_mandi','luas_rumah','lantai','kondisi','Ket','alamat','kota']]
					

					
				btn = st.button("Cari")
				if btn:
					st.write('Prediksi Harga Rumah Rata-Rata **${:.2f}**'.format(PRED))
					st.map(df_models[['lat','lon']])
					st.write('Berdasarkan hasil Pencarian, Telah ditemukan')
					for i in range(len(asli)):
							kota = asli.iloc[i]['kota']
							alamat = asli.iloc[i]['alamat']
							harga = asli.iloc[i]['harga']
							peringkat = bung.iloc[i]['peringkat']
							st.write(' ~ Peringkat Nomer **{}** Ditemukan Dikota **{}** Dengan Alamat **{}** dan dengan Harga **${:.2f}**'.format(peringkat,kota,alamat,harga))
				else:
					pass
					
		else:
			st.warning("salah Password Bro")

if __name__ == '__main__':
	main()
