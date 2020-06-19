import pandas as pd
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.tree import DecisionTreeRegressor

#proje 2 ABD ev fiyatları
# Regresyon

df = pd.read_csv("final_dataa.csv")


df['zindexvalue'] = df['zindexvalue'].str.replace(',', '')
df["zindexvalue"]= df["zindexvalue"].astype(np.int64)

X = df[["bathrooms", "bedrooms","finishedsqft","totalrooms","yearbuilt","zestimate","zindexvalue"]]
y = df.lastsoldprice


X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.2, random_state=111)
rastgele_orman = RandomForestRegressor(n_estimators=25, random_state=2)
rastgele_orman.fit(X_eğitim, y_eğitim)

y_tahmin = rastgele_orman.predict(X_test)
rmse_test = MSE(y_test, y_tahmin)**(1/2)
print("RMSE değeri (Rastgele Orman): {:.2f}".format(rmse_test))

karar_agacı_reg = DecisionTreeRegressor(max_depth=4, random_state=111)
karar_agacı_reg.fit(X_eğitim, y_eğitim)

y_tahmin = karar_agacı_reg.predict(X_test)
rmse_test = MSE(y_test, y_tahmin)**(1/2)
print("RMSE değeri (Karar Ağacı): {:.2f}".format(rmse_test))

önem_dereceleri = pd.Series(data=rastgele_orman.feature_importances_,
                        index= X_eğitim.columns)

önem_dereceleri_sıralı = önem_dereceleri.sort_values()

önem_dereceleri_sıralı.plot(kind='barh', color='darkred')
plt.title('Değişkenlerin Önem Dereceleri')
plt.show()

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################

# Proje 3 Fraud credit card
# Classification

proje3_df = pd.read_csv("creditcard.csv")

normal_alısveris = proje3_df[proje3_df.Class == 0]
sahte_alısveris = proje3_df[proje3_df.Class == 1]

normal_alısveris_azaltılmış = resample(normal_alısveris,
                                     replace = True,
                                     n_samples = len(sahte_alısveris),
                                     random_state = 111)

azaltılmış_df = pd.concat([sahte_alısveris, normal_alısveris_azaltılmış])

X = azaltılmış_df.drop('Class', axis=1)
y = azaltılmış_df['Class']

X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.20, random_state=111)

rastgele_orman = RandomForestClassifier(n_estimators=25, random_state=2)
rastgele_orman.fit(X_eğitim, y_eğitim)

y_tahmin_ro = rastgele_orman.predict(X_test)
print("Rastgele Orman Doğruluk Değeri : {:.2f}".format(accuracy_score(y_test, y_tahmin_ro)))

önem_dereceleri = pd.Series(data=rastgele_orman.feature_importances_,
                        index= X_eğitim.columns)

önem_dereceleri_sıralı = önem_dereceleri.sort_values()

önem_dereceleri_sıralı.plot(kind='barh', color='darkblue')
plt.title('Değişkenlerin Önem Dereceleri')
plt.show()


