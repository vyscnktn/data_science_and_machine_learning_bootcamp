# İş Problemi

# Online ayakkabı mağazası olan FLO müşterilerini
# segmentlere ayırıp bu segmentlere göre pazarlama
# stratejileri belirlemek istiyor. Buna yönelik olarak
# müşterilerin davranışları tanımlanacak ve bu
# davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

# Veri Seti Hikayesi

#Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel
# (hem online hem offline alışveriş yapan)
#olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen
# bilgilerden oluşmaktadır.

# Değişkenler

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin aptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


# Görev 1: Veriyi Anlama ve Hazırlama

# Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

import datetime as dt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)   # bütün sütunlar gözüksün
# pd.set_option('display.max_rows', None)   # bütün satırlar gözüksün
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("3. Hafta CRM Analitik/ödev/flo_data_20k.csv")

# Adım 2: Veri setinde
#         a. İlk 10 gözlem,
#         b. Değişken isimleri,
#         c. Betimsel istatistik,
#         d. Boş değer,
#         e. Değişken tipleri, incelemesi yapınız.

df = df_.copy()
df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.info()


# Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan
# alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz

df["total_shopping"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_shopping"]
df.head()

df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["total_price"]
df.head()

# Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes

pd.to_datetime(df["first_order_date"])
pd.to_datetime(df["last_order_date"])
pd.to_datetime(df["last_order_date_online"])
pd.to_datetime(df["last_order_date_offline"])
df.dtypes

# Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün
# sayısının ve toplam harcamaların dağılımına bakınız.

df.groupby(["order_channel", "total_shopping", "total_price"]).agg({"master_id": "count",
                                                                    "total_shopping": "sum",
                                                                    "total_price": "sum"})


# Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.sort_values(by="total_price", ascending=False).head(10)


# Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.sort_values(by="total_shopping", ascending=False).head(10)

# Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.

def create_flo(dataframe):

    # Veriyi Anlama ve Hazırlama
    dataframe.head()
    dataframe.columns
    dataframe.describe().T
    dataframe.isnull().sum()
    dataframe.info()


    # Her bir müşterinin topla alışveriş sayısı ve toplam harcaması
    dataframe["total_shopping"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_shopping"]
    dataframe.head()

    dataframe["total_price"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe["total_price"]
    dataframe.head()

    # Tarih ifade eden değişkenlerin tipini date'e çevirmek
    # pd.to_datetime(dataframe["first_order_date"])
    # pd.to_datetime(dataframe["last_order_date"])
    # pd.to_datetime(dataframe["last_order_date_online"])
    # pd.to_datetime(dataframe["last_order_date_offline"])

    # Dağılıma bakmak
    dataframe.groupby(["order_channel", "total_shopping", "total_price"]).agg({"master_id": "count",
                                                                        "total_shopping": "sum",
                                                                        "total_price": "sum"})


    return dataframe


create_flo(df)




# Görev 2: RFM Metriklerinin Hesaplanması


# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.



# Recency, Frequency, Monetary

# Recency: Müşterinin yeniliğini, sıcaklığını ifade ediyor. Matematiksel karşılığı;
# Analizin yapıldığı tarih - ilgili müşterinin son satın alma yaptığı tarihtir.

# Frequency: Müşterinin yaptığı toplam satın almadır.

# Monetary: Müşterinin bu yaptığı toplam satın almalar neticesinde bıraktığı toplam parasal değerdir.


df.head()

# bu veri seti içerisindeki en son tarih
df["last_order_date"].max()

# biz burda örneğin 2021-06-01 tarihini analiz yapılan tarih gibi kabul ederiz.
# bu tarih üzerinden Recency'i hesaplarız.
today_date = dt.datetime(2021, 6, 1)
type(today_date)

rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (today_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

# last_date = df.columns.str.contains("date")
# df[last_date].apply(pd.to_datetime("date"))

rfm = df.groupby("customer_id").agg({"last_order_date": lambda date: (today_date - date.max()).days,
                                   "total_shopping": lambda shop: shop.nunique(),
                                   "total_price": lambda total_price: total_price.sum()})






# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.

# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.

# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.

# recency değerini hesaplamak için analiz tarihini maksimum tarihten 2 gün sonrası seçebilirsiniz




