import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv("datasets/diabetes.csv")
df = df_.copy()

#genel resim

df.head()

df.describe().T


#  numerik ve kategorik değişkenlerin bulunması ve analizi

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df[cat_cols].head()
df[num_cols].head()
df.isnull().nunique()

df.groupby("Outcome")[num_cols].mean()
df.groupby(cat_cols)[num_cols].mean()




# aykırı gözlem analizi

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low_limit, up_limit = outlier_thresholds(df,num_cols,q1=0.05,q3=0.95)



 def check_outlier(dataframe, col_name):
     low_limit, up_limit = outlier_thresholds(dataframe, col_name)
     if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
         return True
     else:
         return False





def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

a = []
b = []
for cols in df[num_cols].columns:
    b = grab_outliers(df, cols, True)
    for i in b:
        a.append(i)

a.sort()
# def grab_outliers(dataframe, col_name, index=False):
#     low, up = outlier_thresholds(dataframe, col_name)
#
#     if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
#         print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
#     else:
#         print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
#
#     if index:
#         outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
#         return outlier_index

#grab_outliers(df, num_cols)
#grab_outliers(df, num_cols, True)

# Eksik değer analizi
df.isnull().nunique()

# Tüm değişkenler 1 değeri yazdırdı. df.isnull() dataframe'inde 1 adet eşsiz değer var yani False

# Korelasyon Analizi
df.columns
df["Outcome"].corr(df["Insulin"])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# FEATURE ENGINEERING
# Eksik ve Aykırı değerlerin işlenmesi

glucose = df.loc[df["Glucose"]<=0, "Glucose"].index
insulin = df.loc[df["Insulin"]<=0, "Insulin"].index

for col in df.columns:
    print(col + " : " + str((df[f"{col}"] == 0).sum()))

df.head()

df["Glucose"].replace(0, np.nan, inplace=True)

df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]=df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace({0:np.nan})

df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].apply()

df.apply(lambda x: x.fillna(x.median()))

df["Glucose"]=df['Glucose'].fillna(df['Glucose'].median())

df["Glucose"].isna().sum()


for col in df.columns:
    df[col] = df[col].fillna(df[col].median())

df.isnull().sum()


# Adım 2: Yeni değişkenler oluşturunuz.

df.head()

# -20 zayıf 20-25 normal 25-30 şişman 30+ obez


df.loc[(df['BMI'] < 20), 'NEW_BMI'] = 'thin'
df.loc[(df['BMI'] >=20 ) & (df["BMI"] < 25), 'NEW_BMI'] = 'normal'
df.loc[(df['BMI'] >= 25 ) & (df["BMI"] < 30), 'NEW_BMI'] = 'fat'
df.loc[(df['BMI'] >= 30 ), 'NEW_BMI'] = 'obese'


df["NEW_BMI"].isnull().sum()


# Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
df.head()

# Scaling

rs = RobustScaler()
df["Pregnancies_robust"] = rs.fit_transform(df[["Pregnancies"]])

for col in num_cols:
    df[[col+"_robust"]] = rs.fit_transform(df[[col]])
df.head()

# Model

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
