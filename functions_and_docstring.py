import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

# Görev:  cat_summary() fonksiyonuna1 özellik ekleyiniz.
# Bu özellik argümanla biçimlendirilebilir olsun.
# Var olan özelliği de argümanla kontrol edilebilir hale getirebilirsiniz.

def cat_summary(dataframe, col_name, isnull=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if isnull:
        print(len([nulls for nulls in dataframe[col_name].isnull() if nulls]))
    print("##########################################")

cat_summary(df,"sex", isnull=True)



# Görev: check_df(), cat_summary() fonksiyonlarına 4 bilgi(uygunsa) barındıran numpy tarzı docstring yazınız.
# (task, params, return, example)


def cat_summary(dataframe, col_name, isnull=False):

    """
    Task
    ----------
    Girilen dataframe'in girilen sütunundaki değişkenlerin sayısını ve değişkenlerin yüzde cinsinden oranını yazdırır.
    Ayrıca null eleman sayısını yazdırır.
    Parameters
    ----------
    dataframe: dataframe
        Özetlenmesini istediğiniz dataframe
    col_name: string
        Özetlenmesini istediğiniz dataframe sütunun adı (kategorik değişkenler içermeli).
    isnull: bool, öntanımlı argümanı: False
        girdiğiniz dataframe'in girdiğiniz sütunundaki değerlerde gezinir ve null eleman sayısını verir

    Returns
    -------
    None

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if isnull:
        print("null eleman sayısı:",len([nulls for nulls in dataframe[col_name].isnull() if nulls]))
    print("##########################################")


cat_summary(df,"sex", isnull=True)

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

check_df(df[num_cols])




