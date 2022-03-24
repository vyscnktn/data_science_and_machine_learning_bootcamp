import pandas as pd
import numpy as np
import seaborn as sns


# Soru 1: persona.csvdosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz
df = pd.read_csv("Python/python_for_data_science/Kural_Tabanli_Siniflandirma/Kural_Tabanli_Siniflandirma/persona.csv")


df.info()


def check_df(dataframe, head=5):
    """
    Task
    ----------
        Girilen dataframe hakkında bazı özet bilgileri yazdırır.
        Boyutu, tipi, ilk -head- elemanı, son -head- elemanı, çeyreklik değerleri.

    Parameters
    ----------
    dataframe: dataframe
        Girdiğiniz dataframe bool ifade içermemeli.
    head: integer; default: 5
        Bu argüman girilen dataframe'de baştan ve sondan yazdırılacak eleman sayısını belirtmek için kullanılır.


    Returns
    -------
    None

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Soru 2: Kaç uniqueSOURCE vardır? Frekansları nedir?

df["SOURCE"].nunique()
df["SOURCE"].unique()

df["SOURCE"].value_counts()


# Soru 3:Kaç uniquePRICE vardır?

df["PRICE"].unique()
df["PRICE"].nunique()

# Soru 4:Hangi PRICE'dankaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()


# Soru 5:Hangi ülkeden kaçar tane satış olmuş?§

df.groupby("COUNTRY").agg({"PRICE":"count"})


# Soru 6:Ülkelere göre satışlardan toplam ne kadar kazanılmış?


df.groupby("COUNTRY").agg({"PRICE":"sum"})


# Soru 7:SOURCE türlerine göre satışsayıları nedir?

df.groupby("SOURCE").agg({"PRICE":"count"})


# Soru 8:Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY").agg({"PRICE":"mean"})

# Soru 9:SOURCE'laragöre PRICE ortalamaları nedir?

df.groupby("SOURCE").agg({"PRICE":"mean"})

# Soru 10: COUNTRY-SOURCE kırılımındaPRICE ortalamaları nedir?

df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE":"mean"})

# Görev 2:  COUNTRY, SOURCE, SEX, AGE kırılımındaortalama kazançlar nedir?

df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})

# Görev 3:  ÇıktıyıPRICE’agöre sıralayınız.

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})
agg_df.sort_values("PRICE",ascending=False)

# Görev 4:  Indeksteyer alan isimleri değişken ismine çeviriniz.

agg_df= agg_df.reset_index()

# Görev 5:  Age değişkenini kategorik değişkene çeviriniz ve agg_df’eekleyiniz


bins = [agg_df["AGE"].min(), 20, 25, 30, 40, agg_df["AGE"].max()]


label_list = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=label_list)
agg_df.head()



# Görev 6:  Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# •Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve
#   veri setine değişken olarak ekleyiniz.
# •Yeni eklenecek değişkenin adı: customers_level_based
# •Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based
#   değişkenini oluşturmanız gerekmektedir


for row in agg_df.values:
    print(row)


[row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
agg_df.head()


agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()

for i in agg_df["customers_level_based"].values:
    print(i.split("_"))

# df["customer_level_based"] = [[df["COUNTRY"].iloc[i]+"_"+
#                                    df["SOURCE"].iloc[i]+"_"+
#                                    df["SEX"].iloc[i]+"_"+
#                                    df["AGE_CAT"][i]] for i in df.index]




"""Görev 7:  Yeni müşterileri (personaları) segmentlere ayırınız.
 •Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’agöre 4 segmenteayırınız.
 •SegmentleriSEGMENTisimlendirmesi ile değişken olarak agg_df’eekleyiniz.
 •Segmentleribetimleyiniz (Segmentleregöre groupbyyapıp pricemean, max, sum’larınıalınız)."""

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

agg_df.columns

agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})


# Görev 8:  Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.
# •33 yaşında ANDROID kullanan bir Türk kadını hangi segmenteaittir ve ortalama ne kadar gelir kazandırması beklenir?
# •35 yaşında IOS kullanan bir Fransız kadını hangi segmenteaittir ve ortalama ne kadar gelir kazandırması beklenir?

agg_df.groupby("SEGMENT").agg({"PRICE":"mean"})


new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
