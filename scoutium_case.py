import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import warnings

warnings.simplefilter(action='ignore', category=Warning)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df1_ = pd.read_csv("Python/dsmlbc/machine_learning/Scoutium/scoutium_attributes.csv",sep=";")
df2_ = pd.read_csv("Python/dsmlbc/machine_learning/Scoutium/scoutium_potential_labels.csv",sep=";")

df1 = df1_.copy()
df2 = df2_.copy()



df1.head()
df2.head()

# "task_response_id", "match_id", "evaluator_id", "player_id" değişkenlerine göre birleştirme işlemi yapıyoruz
df1.columns
df2.columns
df = pd.merge(df1,df2, on=["task_response_id", "match_id", "evaluator_id", "player_id"])
df.head()
df.count()

# position_id içerisindeki Kaleci(1) sınıfını verisetinden kaldırınız.
df = df[df["position_id"]!=1]
df["position_id"].unique()
df.count()

# potential_label içerisindeki below_average sınıfını verisetinden kaldırınız.

df = df[df["potential_label"]!="below_average"]
df.count()

# pivot table  index “player_id”,“position_id” ve“potential_label” sütun "attribute_id” değer attribute_value

df.pivot_table(index=["player_id","position_id","potential_label"],columns="attribute_id", values="attribute_value")

df = df.reset_index()

# tip dönüşümü
df.columns
df.dtypes
df["attribute_id"] = df["attribute_id"].apply(str)

# encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
label_encoder(df,"potential_label").head()


# sütunlar

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# scaling

ss = StandardScaler()
for cols in num_cols:
    df[cols] = ss.fit_transform(df[[cols]])

df.head()

# modelin kurulması
df = df[["task_response_id", "match_id", "evaluator_id", "player_id", "position_id", "analysis_id", "attribute_value", "potential_label"]]
df.drop()
y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)
################################################
# XGBoost
################################################
df.dtypes
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.75265
cv_results['test_f1'].mean()
# 0.631
cv_results['test_roc_auc'].mean()
# 0.7987

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

