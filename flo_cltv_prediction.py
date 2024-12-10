# Görev 1:  Veriyi Hazırlama


# Adım1:   flo_data_20K.csv verisiniokuyunuz.


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


df_ = pd.read_csv("Python/dsmlbc/FLO_RFM_Analizi/flo_data_20k.csv")

df = df_.copy()


# Adım2:  Aykırıdeğerleribaskılamakiçingerekliolanoutlier_thresholdsvereplace_with_thresholdsfonksiyonlarınıtanımlayınız.
# Not: cltvhesaplanırkenfrequency değerleriinteger olmasıgerekmektedir.Bunedenlealt veüstlimitleriniround() ileyuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

# Adım3:  "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online" değişkenlerininaykırıdeğerlerivarsabaskılayanız.

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df.describe().T
# Adım4:  Omnichannel müşterilerinhem online'danhem de offline platformlardanalışverişyaptığınıifadeetmektedir.
# Her birmüşterinintoplamalışverişsayısıveharcamasıiçinyeni değişkenleroluşturunuz.


df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["customer_value_total_ever"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df
# Adım5:  Değişkentipleriniinceleyiniz. Tarihifadeedendeğişkenlerintipinidate'eçeviriniz.


df.dtypes

df['first_order_date'] = pd.to_datetime(df['first_order_date'])

df['last_order_date'] = pd.to_datetime(df['last_order_date'])

df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])

df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])

df.dtypes


# Görev 2:  CLTV Veri Yapısının Oluşturulması

# Adım1:  Veri setindekienson alışverişinyapıldığıtarihten2 günsonrasınıanaliztarihiolarakalınız.


df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)
type(today_date)
# Adım2:  customer_id, recency_cltv_weekly, T_we ekly, frequency vemonetary_cltv_avgdeğerlerininyeraldığıyeni
# bircltvdataframe'ioluşturunuz. Monetary değerisatınalma başınaortalamadeğerolarak, recency vetenure değerleri
# isehaftalıkcinstenifadeedilecek

today_date
"""
rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'order_num_total_ever': lambda order_num_total_ever: order_num_total_ever.sum(),
                                     'customer_value_total_ever': lambda customer_value_total_ever: customer_value_total_ever.sum()})


cltv_df = df.groupby('master_id').agg({'last_order_date': [lambda date: (df["last_order_date"].max() - date.min()).days,
                                                         lambda date1: (today_date - date1.max()).days],
                                         'order_num_total_ever': lambda num: num,
                                         'customer_value_total_ever': lambda TotalPrice: TotalPrice.sum()})


cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
"""
cltv_df.columns = cltv_df.columns.droplevel(0)


cltv_df.columns = ["recency", "T", "frequency", "monetary"]

# customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg

cltv_df["monetary_cltv_avg"] = cltv_df["monetary"]/cltv_df["frequency"]

cltv_df = cltv_df[cltv_df["monetary"]>0]

cltv_df["recency_cltv_weekly"] = cltv_df["recency"]/7

cltv_df["T_weekly"] = cltv_df["T"]/7

cltv_df = cltv_df[(cltv_df["frequency"]>1)]

cltv_df.head()



# Görev 3:  BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’ninHesaplanması

# Adım1:  BG/NBD modelinifit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df["recency_cltv_weekly"],
        cltv_df['T_weekly'])



bgf.conditional_expected_number_of_purchases_up_to_time(24,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly']).sort_values(ascending=False).head(10)
"""
bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)"""

cltv_df["expected_purc_6_months"] = bgf.predict(24,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])



cltv_df["expected_purc_3_months"] = bgf.predict(12,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

cltv_df



ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df["monetary_cltv_avg"])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary_cltv_avg']).sort_values(ascending=False).head(10)

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary_cltv_avg'])
cltv_df.sort_values("exp_average_value", ascending=False).head(20)

###########################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="master_id", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(20)

cltv.dtypes

# Görev 4:  CLTV Değerine Göre SegmentlerinOluşturulması

# Adım1:  6 aylıkCLTV'yegöretümmüşterilerinizi4 gruba(segmente) ayırınızvegrupisimleriniverisetineekleyiniz.
# Adım2:  4 grupiçerisindenseçeceğiniz2 grupiçinyönetimekısakısa6 aylıkaksiyonönerilerindebulununuz.


cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final["segment"]

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


cltv_final.index = cltv_final.index.astype(int)

cltv_final["clv"].max()

cltv_final.groupby("segment").agg({"expected_purc_3_months":["mean", "sum"]})
