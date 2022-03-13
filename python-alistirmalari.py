import seaborn as sns

# Görev 1:  Verilen değerlerin veri yapılarını inceleyiniz.


x = 8
y = 3.2
z = 8j + 18
a = "hello world"
b = True
c = 23 < 22
l = [1,2,3,4]
d = {"Name":"Jake",
     "Age":27,
     "Adress":"Downtown"}
t = ("Machine Learning","Data Science")
s = {"Python","Machine Learning","Data Science"}

print(type(x), type(y), type(z), type(a), type(b), type(c), type(l), type(d), type(t), type(s))


# Görev 2:  Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız

text = "The goal is to turn data into information, and information into insight."

TEXT = text.upper()
print(TEXT)

TEXT = TEXT.replace(",","")
TEXT = TEXT.replace(".","")
TEXT = TEXT.split(" ")
print(TEXT)


# Görev 3:  Verilen listeye aşağıdaki adımları uygulayınız.
# Adım1: Verilen listenin eleman sayısına bakınız.
# Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım4: Sekizinci indeksteki elemanı siliniz.
# Adım5: Yeni bir eleman ekleyiniz.
# Adım6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

lst = ["D","A","T","A","S","C","I","E","N","C","E"]

print(len(lst))

print(lst[0],lst[10])

data = lst[0:4]

print(data, type(data))

lst.pop(8)

print(lst)

lst.append("P")
print(lst)

lst.insert(8,"Y")
print(lst)

# Görev 4:  Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
# Adım1: Key değerlerine erişiniz.
# Adım2: Value'lara erişiniz.
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
# Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
# Adım5: Antonio'yu dictionary'den siliniz.

dict = {"Christian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

print(dict.keys())

print(dict.values())

dict["Daisy"][1] = 13
print(dict["Daisy"][1] )

dict["Ahmet"] = ["Turkey",24]

print(dict)

dict.pop("Antonio")

print(dict)



# Görev 5:Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan
# ve bu listeleri return eden fonksiyon yazınız

num = [1,2,3,4,5,6,7,8,9,10]
tek = []
cift = []

def sayi (liste):

    for i in range(len(liste)):
       if liste[i]%2==0:
           cift.append(liste[i])

       else:
           tek.append(liste[i])

    return cift,tek


# Aynı çıktıyı list comprehension ile uygulamak

tek_1 = [i for i in range(len(num)) if not i%2==0]

cift_1 = [i for i in range(len(num)) if i%2==0]

print(tek_1,"\t",cift_1)


# Görev 6:  List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini
# büyük harfe çeviriniz ve başına NUM ekleyiniz

df = sns.load_dataset("car_crashes")

print(df.columns)

a = [col.upper() for col in df.columns]

print(a)

# Görev 7:  List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan değişkenlerin
# isimlerinin sonuna "FLAG" yazınız

b = [i if "NO" in i else i+"_FLAG" for i in a]

print(b)

# Görev 8:  List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden
# FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz

og_list = ["abbrev","no_previous"]

new_cols = [i for i in df.columns if not i==og_list[0] or i==og_list[1]]

print(new_cols)

new_df = df[new_cols]

print(new_df.head())