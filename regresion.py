import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import datetime
import re


df = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Quinto Año\1er Semestre\Top. Esp. Electronica\Proyecto Final\Base de datos_RIO.csv', parse_dates=["date"])
# print(df)
# df = df[0:1000]
time_1 = df['date']
df['dayofyear'] = time_1.dt.dayofyear
# time = np.array(df['dayofyear'])
# time = time.reshape(-1,1)
# time = pd.DataFrame(time)

prom = []
ph_prom = []
days = np.arange(170,280)
for i in days:
       a = df.loc[df['dayofyear'] == i]
       suma = a['turbidity (fnu)'].sum()/len(a['turbidity (fnu)'])
       prom.append(suma)

df_1 = pd.DataFrame()
df_1['ph_prom'] = prom

df_1['dia_prom'] = days
time = np.array(df_1['dia_prom'])
time = time.reshape(-1,1)
time = pd.DataFrame(time)
print(time)
# print(len(prom))
# print(prom)
# print(time)
print(df_1)

# # time_1 = pd.to_datetime(time_1, format='%H:%M:%S')
# formatted = []

# for times in time_1:
#        date_time_obj = datetime.datetime.strptime(str(times), '%m/%d/%Y')
#        hours = date_time_obj.strftime('%Y/%m/%d')
#        formatted.append(hours)
# # time_1 = pd.DataFrame(formatted)


# df['time'] = formatted
# print(df['time'])
# df['dayofyear'] = df['time'].dt.dayofyear
# print(df['dayofyear'])

# time_1 = time_1.values.reshape(-1, 1)



# print(time_1)


# print(time)
# print(ph)
# X = np.sort([[1985],[1986],[1987],[1988],[1989],[1990],[1991],[1992],[1993],[1994],[1995],[1996],[1997],[1998],[1999],[2000],[2001],[2002],[2003],[2004],[2005],[2006],[2007],[2008],[2009],[2010],[2011],[2012],[2013],[2014],[2015],[2016],[2017],[2018],[2019],[2020],[2021]])
# # X = np.sort([[2000],[2001],[2002]])

# X_1 = ([[2000],[2001],[2002],[2003],[2004],[2005],[2006],[2007],[2008],[2009],[2010],[2011],[2012],[2013],[2014],[2015],[2016],[2017],[2018],[2019],[2020],[2021]])
# Y_4 = np.array([4967.50,4999.90,5221.80,5342.60,5571.00,5710.50,5861.00,6209.00,6240.00,6753.70,7290.00,7722.60,8359.80,8722.00,9151.00,9939.00,10278.00,10533.80,10709.00,11116.00,10309.00])

# y = np.array([424,446,475,471,446,464,488,518,541,592,619,640,707,726,754,777,839,857,883,925,946,971,1024,1064,1154,1222,1287,1386,1443,1504,1612,1618,1657,1665,1961,1969,2020])
# y_3 = ([777,839,857,883,925,946,971,1024,1064,1154,1222,1287,1386,1443,1504,1612,1618,1657,1665,1961,1969,2020])

# X = np.sort(2000 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()

# #############################################################################
# # Add noise to targets
# y[::5] += 3 * (0.5 - np.random.rand(8)) 

# print(X)
# print(y)


# X_train, X_test, y_train, y_test = train_test_split(
#                         time, ph,
#                 test_size = 0.20, random_state = 42)
# print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))


reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()

regr_1 = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
regr_2 = SVR(kernel="rbf")
# regr_1.fit(X_1, y_3)
# regr_2.fit(X_1, y_3)
regr_2.fit(time,df_1['ph_prom'])
# print('Precisión SVM: {}'.format(regr_2.score(X_train, y_train)))
# print('Precisión SVM testeo: {}'.format(regr_2.score(X_test, y_test)))
# regr_2.fit(time_1, ph)
print(regr_2.predict([[190], [200], [220]]))

# Predict
X_test = np.arange(2022, 2052, 1)[:, np.newaxis]
X_test_L = np.arange(170, 300, 1)[:, np.newaxis]
X_test1 = np.arange(2022, 2052, 1)
y_1 = regr_2.predict(X_test_L)
# y_2 = regr_2.predict(X_test)

# y_1L = regr_1.predict(X_test_L)
# y_2L = regr_2.predict(X_test_L)

# print(y_2)
salto = np.arange(2000, 2052, step=1)

y1err = 1 + 1 * np.sqrt(X_test1)

# Plot the results
plt.figure()
plt.plot(time, df_1['ph_prom'], label="Datos en tiempo real")

# plt.errorbar(X_test, y_1, yerr=y1err, color="cornflowerblue", label="Regresor de Votacion", linewidth=2)
plt.plot(X_test_L, y_1,linewidth=2, label ='Prediccion a futuro')

# plt.errorbar(X_test, y_2,yerr=y1err,color="yellowgreen", label="Regresion Lineal", linewidth=2)
# plt.plot(X_test_L, y_2L,color="yellowgreen", linewidth=2)

plt.xlabel("Dias")
plt.ylabel("Nivel de Turbidez")
plt.title("Prediccion de Niveles de Turbidez y Sedimentos en Rio")
plt.legend()
# plt.xticks(salto, salto,
#        rotation=90)
plt.grid(which='major', axis='both', color='0.8', linestyle='-')
plt.show()

# svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
# svr_lin = SVR(kernel="linear", C=100, gamma="auto")
# svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)



# lw = 2

# svrs = [svr_rbf, svr_lin, svr_poly]
# kernel_label = ["RBF", "Linear", "Polynomial"]
# model_color = ["m", "c", "g"]

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
# for ix, svr in enumerate(svrs):
#     axes[ix].plot(
#         X,
#         svr.fit(X, y).predict(X),
#         color=model_color[ix],
#         lw=lw,
#         label="{} model".format(kernel_label[ix]),
#     )
#     axes[ix].scatter(
#         X[svr.support_],
#         y[svr.support_],
#         facecolor="none",
#         edgecolor=model_color[ix],
#         s=50,
#         label="{} support vectors".format(kernel_label[ix]),
#     )
#     axes[ix].scatter(
#         X[np.setdiff1d(np.arange(len(X)), svr.support_)],
#         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
#         facecolor="none",
#         edgecolor="k",
#         s=50,
#         label="other training data",
#     )
#     axes[ix].legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.1),
#         ncol=1,
#         fancybox=True,
#         shadow=True,
#     )

# # fig.text(0.5, 0.04, "data", ha="center", va="center")
# # fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
# fig.suptitle("Support Vector Regression", fontsize=14)
# plt.show()