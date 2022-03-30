# Görev 1:  Veriyi Anlama ve Hazırlama

# Adım1:   flo_data_20K.csv verisiniokuyunuz.Dataframe’inkopyasınıoluşturunuz.

import pandas as pd
import numpy as np
import datetime as dt


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("Python/dsmlbc/FLO_RFM_Analizi/flo_data_20k.csv")

df = df_.copy()

# Adım2:   Veri setinde
#   a. İlk 10 gözlem,
#   b. Değişkenisimleri,
#   c. Betimselistatistik,
#   d. Boşdeğer,
#   e. Değişkentipleri, incelemesiyapınız.

df.head(10)

df.columns

df.describe().T

df.isnull().sum()

df.dtypes

# Adım3:Omnichannel müşterilerinhem online'danhemdeoffline platformlardanalışverişyaptığınıifadeetmektedir.
# Her birmüşterinintoplamalışverişsayısıveharcamasıiçinyeni değişkenleroluşturunuz.

df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["customer_value_total_ever"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()

# Adım4:  Değişkentipleriniinceleyiniz. Tarihifadeedendeğişkenlerintipinidate'eçeviriniz.

df.dtypes

df['first_order_date'] = pd.to_datetime(df['first_order_date'])

df['last_order_date'] = pd.to_datetime(df['last_order_date'])

df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])

df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])

df.dtypes



# Adım5:  Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının
# ve toplam harcamaların dağılımına bakınız.

df.head()
df.groupby("order_channel").agg({"customer_value_total_ever": "sum","order_num_total_ever":"sum"})

df.groupby("master_id").agg({"order_num_total_ever":"sum",
                             "customer_value_total_ever":"sum"})
# Adım6:  En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df["customer_value_total_ever"].sort_values(ascending=False).head(10)

# Adım7:  Enfazlasiparişiverenilk 10 müşteriyisıralayınız.

df["order_num_total_ever"].sort_values(ascending=False).head(10)

# Adım8:  Veri önhazırlıksürecinifonksiyonlaştırınız.

def data_pre_prop(df):
    df.head(10)
    df.describe().T

    df.isnull().sum()
    df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total_ever"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df.groupby("order_channel").agg({"customer_value_total_ever": "sum", "order_num_total_ever": "sum"})
    df["customer_value_total_ever"].sort_values(ascending=False).head(10)
    df["order_num_total_ever"].sort_values(ascending=False).head(10)

df.head()
data_pre_prop(df)


# Görev 2:  RFM Metriklerinin Hesaplanması

# Adım 1: Recency, Frequencyve Monetary tanımlarını yapınız.

df["last_order_date"].max()

today_date = dt.datetime(2021, 5, 30)
type(today_date)

df.head()
# Adım 2: Müşteri özelinde Recency,
# Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfmisimli bir değişkene atayınız.
# Adım 4: Oluşturduğunuz metriklerin isimlerini  recency, frequencyve monetaryolarak değiştiriniz.

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'order_num_total_ever': lambda order_num_total_ever: order_num_total_ever.sum(),
                                     'customer_value_total_ever': lambda customer_value_total_ever: customer_value_total_ever.sum()})

rfm

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T
rfm["monetary"].min()


# Görev 3

#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
#Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
#Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

#Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.head()

rfm[rfm["RFM_SCORE"]=="15"]


# görev 4
#Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
#Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm.head()


#Görev 5:  Aksiyon Zamanı !
# Adım1:  Segmentlerinrecency, frequnecyvemonetary ortalamalarınıinceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# Adım2:  RFM analiziyardımıylaaşağıdaverilen 2 case için ilgili profildekimüşterileribulunve
# müşteriid'lerinicsv olarakkaydediniz.a.FLObünyesineyenibirkadınayakkabımarkasıdahilediyor.
# Dahilettiğimarkanınürünfiyatlarıgenelmüşteritercihlerininüstünde.
# Bunedenlemarkanıntanıtımıveürünsatışlarıiçinilgilenecekprofildekimüşterilerleözelolarakiletişimegeçmekisteniliyor.
# Sadıkmüşterilerinden(champions,loyal_customers)vekadınkategorisindenalışverişyapankişilerözelolarakiletişimkurulacakmüşteriler.
# Bumüşterilerinidnumaralarınıcsvdosyasınakaydediniz.

customer = rfm[(rfm["segment"]== "champions") | (rfm["segment"]=="loyal_customers")]

df[df["interested_in_categories_12"].str.contains("KADIN")]

lst1 = [customer.index]


lst1 = []
for id in df["master_id"]:
    for index in customer.index:
        if id==index:
           lst1.append(id)

lst1

customer= pd.DataFrame(lst1)

customer.to_csv("loyal_customer.csv")

"""set1 = rfm[(rfm["segment"]== "champions") | (rfm["segment"]=="loyal_customers") & (df["interested_in_categories_12"].str.contains("KADIN"))]


set2=df[df["interested_in_categories_12"].str.contains("KADIN")]
son_set = set1.merge(set2, on="master_id", how="left")
son_set["master_id"].index
son_set.to_csv("son_set.csv")
"""


"""df.loc["master_id",00016786-2f5a-11ea-bb80-000d3a38a36f"]

for id in df["master_id"]:
    for index in customer.index:
        if id==index
"""

# b.Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır.
# Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama
# uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler,
# uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor.
# Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.


rfm["segment"].unique()

["about_to_sleep", 'new_customers', 'cant_loose']

rfm2 = rfm[(rfm["segment"]== "about_to_sleep") |
    (rfm["segment"]== "new_customer") |
    (rfm["segment"]== "cant_loose" )]

df2 = df[(df["interested_in_categories_12"].str.contains("COCUK")) |
         (df["interested_in_categories_12"].str.contains("ERKEK"))]
lst = []
for id in df2["master_id"]:
    for index in rfm2.index:
        if id==index:
           lst.append(id)

customer323 = pd.DataFrame(lst)

customer323.to_csv("customer_campaign.csv")