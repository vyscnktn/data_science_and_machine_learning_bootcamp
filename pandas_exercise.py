import numpy as np
import pandas as pd
import seaborn as sns

# Görev 1:  Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.

df = sns.load_dataset("titanic")

# Görev 2:  Titanic verisetindeki kadın ve erkek yolcuların sayısını bulunuz.

df.columns
df["sex"].value_counts()

male = (df["sex"].value_counts())[0]
female = (df["sex"].value_counts())[1]

print("Male:", male, "Female:", female)

# Görev3:  Her bir sutuna ait unique değerlerin sayısını bulunuz

df["sex"].nunique()

unique_count = {col: df[col].nunique() for col in df.columns}

print(unique_count)

# Görev4:  pclass değişkeninin unique değerlerinin sayısını bulunuz

df.pclass.nunique()

# Görev5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz

df[["pclass", "parch"]].nunique()

# Görev6:  embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

df.embarked.dtype

df.embarked = df.embarked.astype("category")

df["embarked"].dtype

# Görev7:  embarked değeriC olanlarıntümbilgelerinigösteriniz

df[df["embarked"] == "C"]
df.iloc[[i for i in range(len(df.embarked)) if df["embarked"].iloc[i] == "C"]]

x = 0
for i, index in enumerate(df.embarked, start=0):
    if i == "C":
        index.append(x)

    x = x + 1

df.iloc[index]

# Görev8:  embarked değeriS olmayanlarıntümbilgelerinigösteriniz

df.iloc[[i for i in range(len(df.embarked)) if df["embarked"].iloc[i] != "S"]]

# Görev9:   Yaşı30 dan küçükvekadınolanyolcularıntümbilgilerinigösteriniz

df["age"].fillna(0, inplace=True)
df.sex.fillna(0, inplace=True)

df.iloc[[i for i in range(len(df["age"])) if ((df.age.iloc[i] <= 30) and (df.sex.iloc[i] == "female"))]]

# Görev10:  Fare'i500'den büyükveyayaşı70’den büyükyolcularınbilgilerinigösteriniz.
df.fare.fillna(0, inplace=True)
df.iloc[[i for i in range(len(df.fare)) if ((df.fare.iloc[i] > 500) & (df.age.iloc[i] > 70))]]

# Görev 11:  Her bir değişkendeki boşdeğerlerintoplamınıbulunuz.
df1 = sns.load_dataset("titanic")
counter = 0
for col in df1.columns:
    print(col)
    for ln in df1[col].isna():
        print(ln)
        if ln:
            counter = counter + 1
print(counter)

# Görev 12:  who değişkeninidataframe’dençıkarınız.

df.columns
df[[i for i in df.columns if i != "who"]]

# Görev13:  deck değikenindekiboşdeğerlerideck değişkeninençoktekraredendeğeri(mode) iledoldurunuz
df1["deck"].mode()
df1["deck"].fillna((df["deck"].mode())[0])

# Görev14:  age değikenindekiboşdeğerleriage değişkeninmedyanıiledoldurunuz

df1["age"].median()
df1["age"].fillna((df["age"].median()))

# Görev15:  survived değişkenininpclassvecinsiyetdeğişkenlerikırılımınındasum, count, mean değerlerinibulunuz

print(df.groupby("survived").agg({"pclass": ["mean", "count"]}))

print(df.groupby("sex").agg({"survived": ["mean", "count"]}))

# Görev16:  30 yaşınaltındaolanlar1, 30'a eşitveüstündeolanlara0 verecekbirfonksiyonyazın.
# Yazdığınızfonksiyonukullanaraktitanikverisetindeage_flagadındabirdeğişkenoluşturunuzoluşturunuz.
# (apply velambda yapılarınıkullanınız)

df["age"].fillna((df["age"].mean()), inplace=True)

df["age_flag"] = df["age"].apply(lambda x: int(x < 30))

# Görev17:  Seaborn kütüphanesi içerisinden Tipsveri setini tanımlayınız

df2 = sns.load_dataset("tips")

# Görev18:  Time değişkenininkategorilerine(Dinner, Lunch) göretotal_billdeğerininsum, min, max vemean değerlerinibulunuz

df2.time.unique()
df2.groupby("time")["total_bill"].agg(["min", "max", "mean"])

# Görev19: Day vetime’agöretotal_billdeğerlerininsum, min, max vemean değerlerinibulunuz.

df2.groupby(["day", "time"])["total_bill"].agg(["min", "max", "mean"])

# Görev 20:  Lunch zamanınavekadınmüşterilereaittotal_billvetip  değerlerininday'egöresum, min, max vemean değerlerinibulunuz.

df2.columns

df2["time"]== "Lunch"
df2[(df2["time"]== "Lunch") & (df2["sex"]== "Female")].groupby("day").agg({"total_bill":["min","max","mean","sum"],
                                                                           "tip":["min","max","mean","sum"]})

df2.loc[((df2["time"]== "Lunch") & (df2["sex"]== "Female")),:].groupby("day").agg({"total_bill":["min","max","mean","sum"],
                                                                           "tip":["min","max","mean","sum"]})


df2.groupby(["time", "sex"])["total_bill"].agg(["sum", "min", "max", "mean"])

# Görev 21:size'i3'ten küçük, total_bill'i10'dan büyükolansiparişlerinortalamasınedir? (loc kullanınız)

df2.loc[[i for i in range(len(df2["size"])) if ((df2["size"].iloc[i]<3) & (df2["total_bill"].iloc[i]>10))],"total_bill"].mean()

# Görev22:  total_bill_tip_sumadındayeni birdeğişkenoluşturunuz. Her birmüşterininödediğitotalbillvetip in toplamınıversin

df2["total_bill_tip_sum"] = df2["total_bill"] + df2["tip"]

df2["total_bill_tip_sum"]
# Görev23:  Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz.
# Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturunuz.
# Kadınlar için Female olanlarının ortalamaları, erkekler için ise Male olanların ortalamaları dikkate alınacktır.
# Parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayınız. (If-else koşullarıiçerecek)

df2 = sns.load_dataset("tips")


female_mean = df2.loc[df2["sex"]=="Female","total_bill"].mean()

male_mean = df2.loc[df2["sex"]=="Male","total_bill"].mean()

def total_bill_flag(sex, total_bill):
    if sex=="Female":
        if total_bill>=female_mean:
            return 1
        else:
            return 0
    else:
        if total_bill>=male_mean:
            return 1
        else:
            return 0

total_bill_flag(df2["sex"][0],df2["total_bill"][0])

df2["total_bill_flag"]= df2.apply(lambda x: total_bill_flag(x["sex"],x["total_bill"]),axis=1)





new = np.arange(0,stop=len(df2.total_bill))
df2["total_bill_flag"] = new

def df2_total_bil(a):
    return df2["total_bill"].iloc[a]

def df2_sex(a):
    return df2["sex"].iloc[a]

def df2_total_bill_flag(a):
    return df2["total_bill_flag"].iloc[a]


male_index = [i for i in range(len(df2.sex)) if df2_sex(i)=="Male"]
female_index = [i for i in range(len(df2.sex)) if df2_sex(i)=="Female"]



mean_male = df2_total_bil(male_index).mean()
mean_female = df2_total_bil(female_index).mean()



for i in range(len(df2.sex)):
    if (df2_sex(i)=="Male"):
        if df2_total_bil(i)<mean_male:
            df2["total_bill_flag"].iloc[i] = 0
        else:
            df2["total_bill_flag"].iloc[i] = 1
    else:
        if df2_total_bil(i)<mean_female:
            df2["total_bill_flag"].iloc[i] = 0
        else:
            df2["total_bill_flag"].iloc[i] = 1


df2["total_bill_flag"] = df2["total_bill"].iloc[[i for i in range(len(df2.sex)) if df2.sex.iloc[i]=="Male"]].apply(lambda x: int(x <(df2["total_bill"].iloc[[i for i in range(len(df2.sex)) if df2.sex.iloc[i]=="Male"]])))

# görev 24

df2.groupby(["sex","total_bill_flag"]).agg({"total_bill_flag":"count"})



# görev 25


df2["total_bill_tip_sum"].sort_values(ascending=False)

df2.columns


df_new = df2.sort_values("total_bill_tip_sum", ascending=False)

df_new