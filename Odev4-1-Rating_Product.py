import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import datetime as dt
import math


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df=pd.read_csv("Python/dsmlbc/Odevler/amazon_review.csv")

df.shape
df.columns
df.head()

# G�rev 1: Average Rating�i g�ncel yorumlara g�re hesaplay�n�z ve var olan average rating ile k�yaslay�n�z.

# Payla��lan veri setinde kullan�c�lar bir �r�ne puanlar vermi� ve yorumlar yapm��t�r. Bu g�revde amac�m�z verilen puanlar� tarihe g�re
# a��rl�kland�rarak de�erlendirmek. �lk ortalama puan ile elde edilecek tarihe g�re a��rl�kl� puan�n kar��la�t�r�lmas� gerekmektedir.

# Ad�m 1: �r�n�n ortalama puan�n� hesaplay�n�z.4.587589013224822

df["overall"].mean()  # 4.587589013224822

# Ad�m 2: Tarihe g�re a��rl�kl� puan ortalamas�n� hesaplay�n�z.

# � reviewTime de�i�kenini tarih de�i�keni olarak tan�tman�z
# � reviewTime'�n max de�erini current_date olarak kabul etmeniz
# � her bir puan-yorum tarihi ile current_date'in fark�n� g�n cinsinden ifade ederek yeni de�i�ken olu�turman�z
# ve g�n cinsinden ifade edilen
# de�i�keni quantile fonksiyonu ile 4'e b�l�p (3 �eyrek verilirse 4 par�a ��kar) �eyrekliklerden
# gelen de�erlere g�re a��rl�kland�rma yapman�z
# gerekir. �rne�in q1 = 12 ise a��rl�kland�r�rken 12 g�nden az s�re �nce yap�lan
# yorumlar�n ortalamas�n� al�p bunlara y�ksek a��rl�k vermek gibi.

df.dtypes
df=df.astype({"reviewTime":"datetime64[ns]"})

current_Date=df["reviewTime"].max()

df.sort_values("day_diff").head(10)

df["fark"] = (current_Date-df["reviewTime"]).dt.days  # burada .dt.days ile integer bir degere cevirdik yoksa 140days 00 hours gibi sacma seyler vard�.
df.shape
df["fark"].dtypes

df.describe().T

a=df["fark"].quantile([0.25,0.5,0.75])


c1= df.loc[(df["fark"]<280.0),"overall"].mean() #4.694762684124386

c2= df.loc[(df["fark"]>=280) & (df["fark"]<430),"overall"].mean()  # 4.639024390243902

c3= df.loc[(df["fark"]>=430) & (df["fark"]<600),"overall"].mean()  # 4.570616883116883

c4= df.loc[(df["fark"]>=600),"overall"].mean()   # 4.446791226645004

c1 * .29 + c2 * .27 + c3 * .26 + c4 * (1-(.29+.27+.26))   # 4.602800574168415

# Ad�m 3: A��rl�kland�r�lm�� puanlamada her bir zaman diliminin ortalamas�n� kar��la�t�r�p yorumlay�n�z.

c1 * .29                # 1.2525365853658537
c2 * .27                # 1.2525365853658537
c3 * .26                #1.1883603896103896
c4 * (1-(.29+.27+.26))  #  0.8004224207961005




# G�rev 2: �r�n i�in �r�n detay sayfas�nda g�r�nt�lenecek 20 review�i belirleyiniz.


# Ad�m 1: helpful_no de�i�kenini �retiniz.
# � total_vote bir yoruma verilen toplam up-down say�s�d�r.
# � up, helpful demektir.
# � Veri setinde helpful_no de�i�keni yoktur, var olan de�i�kenler �zerinden �retilmesi gerekmektedir.
# � Toplam oy say�s�ndan (total_vote) yararl� oy say�s� (helpful_yes) ��kar�larak yararl� bulunmayan oy say�lar�n� (helpful_no) bulunuz.

df["helpful_no"] = df["total_vote"]-df["helpful_yes"]



# Ad�m 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlar�n� hesaplay�p veriye ekleyiniz.

# � score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlar�n� hesaplayabilmek i�in score_pos_neg_diff,
# score_average_rating ve wilson_lower_bound fonksiyonlar�n� tan�mlay�n�z.
# � score_pos_neg_diff'a g�re skorlar olu�turunuz. Ard�ndan; df i�erisinde score_pos_neg_diff ismiyle kaydediniz.
# � score_average_rating'a g�re skorlar olu�turunuz. Ard�ndan; df i�erisinde score_average_rating ismiyle kaydediniz.
# � wilson_lower_bound'a g�re skorlar olu�turunuz. Ard�ndan; df i�erisinde wilson_lower_bound ismiyle kaydediniz

df.head()

def score_pos_neg_diff(df):
    return df["helpful_yes"] - df["helpful_no"]

df["helpful_yes"].isnull().sum()
df["helpful_yes"].count()

def score_average_rating(df):

    for index, i in enumerate(df["total_vote"]):
        if i == 0:
            print(0)
        else:
            return ["helpful_yes"] / (df["total_vote"]


df.loc[4817]

if (df.loc[df["total_vote"] == 0], "total_vote"):
    return (df.loc[df["total_vote"] == 0], "total_vote")




type((df["helpful_yes"]) / (df["total_vote"]))

score_average_rating(df.head(30))

score_average_rating(df)

0 in df["total_vote"]


def wilson_lower_bound(df, confidence=0.95):

        if df["total_vote"] == 0 :
            return 0
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        phat = 1.0 * df["helpful_yes"] / df["total_vote"]
        return (phat + z * z / (2 * df["total_vote"]) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * df["total_vote"])) / df["total_vote"])) / (1 + z * z / df["total_vote"])

def wilson_lower_bound(up, down, confidence=0.95):

    n = up + down
        if n == 0:
            return 0
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        phat = 1.0 * up / n
        return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(df)