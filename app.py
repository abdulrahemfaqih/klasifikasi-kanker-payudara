import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import math

# fungsi
from functions import penjelasan_fitur
from functions import plot_distribusi_kelas
from functions import plot_distribusi_fitur_kelas
from functions import menampilkan_informasi_data
from functions import menampilkan_outlier
from functions import tampilkan_heatmap_korelasi
from functions import handle_outliers
from functions import penjelasan_IQR

# data
from data import kolom_fitur

# st.set_page_config(layout="wide")
st.title(
    "Perbandingan Model ANN, Logistic Regression dan SVM pada Klasifikasi Kanker Payudara"
)
st.write(
    "**220411100029 | Abdul Rahem Faqih | Proyek Sains Data IF 5D | Teknik Informatika**"
)
data_understanding, data_prepocessing, modelling_evaluasi, implementasi = st.tabs(
    ["Data Understanding", "Preprocessing", "Modeling", "Implementasi"]
)


def load_data():
    df = pd.read_csv("breast_cancer.csv")
    return df


with data_understanding:
    st.write(
        "Data yang digunakan adalah data kanker payudara yang diambil dari [kaggle](https://www.kaggle.com/). Data ini memiliki 569 baris dan 33 kolom. Kolom diagnosis adalah target yang akan diprediksi. Kolom diagnosis memiliki dua kelas yaitu M (Malignant) dan B (Benign)."
    )
    # pratinjau dataset
    st.subheader("Pratinjau Dataset")
    df = load_data()
    st.table(df.head())
    df = df.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")

    # penjelesan fitur
    st.header("Penjelasan Fitur")
    st.write(
        "Dengan menjelaskan setiap fitur, kita dapat memahami karakteristik data yang akan digunakan dalam analisis dan pemodelan."
    )
    df_fitur = pd.DataFrame(
        {
            "Fitur": kolom_fitur,
            "Penjelasan": [penjelasan_fitur(fitur) for fitur in kolom_fitur],
        }
    )
    st.table(df_fitur)
    # Menampilkan distribusi setiap kelas dalam bentuk bar chart

    st.subheader("Distribusi Kelas")
    plot_distribusi_kelas(df)
    st.write(
        "Dapat dilihat bahwa distribusi label data tidak seimbang antara kelas Benign dan Malignant. Oleh karena itu, pada tahap preprocessing perlu dilakukan penyeimbangan data training agar model tidak memiliki bias terhadap kelas mayoritas."
    )

    st.subheader("Distribusi Fitur terhadap Kelas")
    # mengubah kelas ke dalam bentuk numerik
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    plot_distribusi_fitur_kelas(df, "diagnosis")

    # menampilkan informasi data
    st.subheader("Informasi Data")
    menampilkan_informasi_data(df)
    st.write(
        "karena untuk kolomnya tidak ada yang memiliki missing value, maka tidak perlu dilakukan handling missing value."
    )

    # menampilkan pengecekan outlier
    st.subheader("PengecekekanOutlier menggunakan IQR")
    st.write(
        "ouutlier adalah data yang berada jauh dari data lainnya. Outlier dapat mempengaruhi hasil analisis data. Outlier dapat diidentifikasi dengan menggunakan metode IQR (Interquartile Range)."
    )
    menampilkan_outlier(df)

    # menampilkan korelasi fitur
    st.subheader("Melihat korelasi fitur menggunakan correlation matrix")
    tampilkan_heatmap_korelasi(df)
    st.write(
        "Dari heatmap korelasi di atas, dapat dilihat bahwa beberapa fitur memiliki korelasi yang tinggi. Korelasi yang tinggi antara fitur dapat mempengaruhi performa model. Oleh karena itu, pada tahap preprocessing perlu dilakukan pemilihan fitur menggunakan ambang batas korelasi terhadap target yang akan digunakan dalam pemodelan."
    )


with data_prepocessing:
    st.markdown(
        """
    Pada **tahap preprocessing** ini, kita akan melakukan serangkaian langkah penting untuk mempersiapkan data sebelum digunakan dalam model. Berikut adalah langkah-langkah yang akan dilakukan:"""
    )

    st.subheader("Langkah-langkah Preprocessing")

    # PENANGANAN OUTLIER
    st.subheader("1. **Penanganan Outlier**")
    st.write(
        "dapat dilihat pada bagian data understanding terdapat outlier pada beberapa fitur, maka pada tahap ini akan dilakukan handling outlier."
    )
    st.write("untuk kasus ini akan digunakan metode IQR (Interquartile Range).")
    penjelasan_IQR()

    data = load_data()
    data = data.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    kolom_numerik = data.select_dtypes(include=["float64", "int64"]).columns
    data_sebelum = data.copy()

    for kolom in kolom_numerik:
        data = handle_outliers(data, kolom)

    col1, col2 = st.columns(2)
    Q1 = data_sebelum.quantile(0.25)
    Q3 = data_sebelum.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    with col1:
        st.write("Jumlah Oulier pada setiap fitur sebelum handling outlier")
        outliers = (data_sebelum < lower_bound) | (data_sebelum > upper_bound)
        outlier_counts = outliers.sum()
        st.table(outlier_counts)
    with col2:
        st.write("Jumlah Oulier pada setiap fitur setelah handling outlier")
        outliers = (data < lower_bound) | (data > upper_bound)
        outlier_counts = outliers.sum()
        st.table(outlier_counts)

    st.write(
        "dapat dilihat perbandingan 2 tabel diatas yaitu yang sebelum dan yang sesudah, pada tabel yang sebelum dilakukan masih terdeteksi outlier, sedangkan pada tabel yang sesudah tidak terdeteksi outlier."
    )

    # SELEKSI FITUR
    st.subheader("2. **Seleksi Fitur**")
    st.write(
        "tahap yang dilakukan setelahnya yaitu adalah pemilihan fitur yang paling berkorelasi dengan target., karena data ini terdapat 33 fitur, maka akan dilakukan pemilihan fitur yang paling berkorelasi dengan target."
    )
    correlations_spearman = data.corr("spearman")["diagnosis"].sort_values(
        ascending=False
    )
    st.table(correlations_spearman)
    st.write(
        "Dari tabel di atas, dapat dilihat bahwa beberapa fitur memiliki korelasi yang tinggi dengan target. Pada tahap ini, kita akan memilih fitur yang memiliki korelasi dengan target di atas ambang batas 0.6"
    )

    correlations_spearman = data.corr("spearman")["diagnosis"].sort_values(
        ascending=False
    )
    significant_features = correlations_spearman[
        correlations_spearman.abs() > 0.6
    ].index.tolist()
    significant_features.remove("diagnosis")

    st.table(pd.DataFrame({"Fitur yang Dipilih": significant_features}))

    st.write("3. **Pembagian Data**")
    st.write(
        "Setelah dilakukan pemilihan fitur di atas, tahap selanjutnya adalah pembagian dataset menjadi data training dan data testing. Data training akan digunakan untuk melatih model, sedangkan data testing akan digunakan untuk menguji performa model yang telah dilatih."
    )

    X = data[significant_features]
    y = data["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.write("Jumlah Data Training:", len(X_train))
    st.write("Jumlah Data Testing:", len(X_test))
    st.write(
        "Persentase Data Training:",
        math.ceil(round(len(X_train) / len(X) * 100, 2)),
        "%",
    )
    st.write(
        "Persentase Data Testing:",
        math.floor(round(len(X_test) / len(X) * 100, 2)),
        "%",
    )

    st.write(
        "Setelah pembagian dataset, tahap selanjutnya adalah normalisasi data. Normalisasi data dilakukan untuk mengubah skala data sehingga memiliki rentang nilai yang sama. Hal ini penting untuk beberapa algoritma machine learning seperti SVM dan KNN."
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write("Data Training setelah normalisasi:")
    st.table(pd.DataFrame(X_train_scaled, columns=significant_features).head())

    st.write("Data Testing setelah normalisasi:")
    st.table(pd.DataFrame(X_test_scaled, columns=significant_features).head())

    st.write(
        "Terakhir, tahap penyeimbangan data dilakukan untuk mengatasi ketidakseimbangan jumlah sampel antara kelas benign dan malignant. Pada tahap ini, akan digunakan metode SMOTE (Synthetic Minority Over-sampling Technique) untuk membuat sampel sintetis dari kelas minoritas."
    )

    oversampler = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = oversampler.fit_resample(
        X_train_scaled, y_train
    )

    y_train_visual_balanced = pd.Series(y_train_balanced).map(
        {0: "Benign", 1: "Malignant"}
    )
    y_train_visual = pd.Series(y_train).map({0: "Benign", 1: "Malignant"})
    y_test_visual = pd.Series(y_test).map({0: "Benign", 1: "Malignant"})
    st.write("Distribusi kelas sebelum SMOTE:")
    st.write("Training set:")
    st.table(pd.Series(y_train_visual).value_counts())
    st.write("Testing set:")
    st.table(pd.Series(y_test).value_counts())

    st.write("Distribusi kelas setelah SMOTE:")
    st.write("Training set:")
    st.table(pd.Series(y_train_visual_balanced).value_counts())
    st.write("Testing set:")
    st.table(pd.Series(y_test).value_counts())
    st.write(
        "Setelah tahap preprocessing selesai, data siap untuk digunakan dalam tahap pemodelan."
    )

    st.session_state["X_train"] = X_train_balanced
    st.session_state["y_train"] = y_train_balanced
    st.session_state["X_test"] = X_test_scaled
    st.session_state["y_test"] = y_test

with modelling_evaluasi:
    if "X_train" not in st.session_state:
        st.error("Data belum diproses. Silakan lakukan preprocessing terlebih dahulu.")
    else:
        st.write(
            "Pada tahap ini, kita akan membandingkan performa tiga model machine learning yang berbeda yaitu Artificial Neural Network (ANN), Logistic Regression, dan Support Vector Machine (SVM) dalam melakukan klasifikasi kanker payudara."
        )

        st.write("1. **Logistic Regression**")
        st.write(
            "Logistic Regression adalah model regresi yang digunakan untuk memprediksi probabilitas dari variabel target biner. Pada kasus ini, kita akan menggunakan Logistic Regression untuk memprediksi apakah tumor payudara bersifat benign atau malignant."
        )

        model_lr = LogisticRegression(random_state=42)
        model_lr.fit(st.session_state["X_train"], st.session_state["y_train"])
        y_pred_lr = model_lr.predict(st.session_state["X_test"])
        accuracy_lr = accuracy_score(st.session_state["y_test"], y_pred_lr)
        st.write("Akurasi Logistic Regression:", accuracy_lr)
        st.write("Classification Report Logistic Regression:")
        class_report_lr = classification_report(st.session_state["y_test"], y_pred_lr)
        st.text(class_report_lr)
        st.write("Confusion Matrix Logistic Regression:")
        cm_lr = confusion_matrix(st.session_state["y_test"], y_pred_lr)
        cm_lr_df = pd.DataFrame(
            cm_lr, index=["True Class 0", "True Class 1"], columns=["Pred 0", "Pred 1"]
        )
        st.dataframe(cm_lr_df)

        st.write("2. **Support Vector Machine (SVM)**")
        st.write(
            "Support Vector Machine (SVM) adalah model machine learning yang digunakan untuk klasifikasi dan regresi. SVM mencari hyperplane terbaik yang memisahkan dua kelas data. Pada kasus ini, kita akan menggunakan SVM untuk memprediksi apakah tumor payudara bersifat benign atau malignant."
        )

        model_svm = SVC(random_state=42)
        model_svm.fit(st.session_state["X_train"], st.session_state["y_train"])
        y_pred_svm = model_svm.predict(st.session_state["X_test"])
        accuracy_svm = accuracy_score(st.session_state["y_test"], y_pred_svm)
        st.write("Akurasi Support Vector Machine:", accuracy_svm)
        st.write("Classification Report Support Vector Machine:")
        class_report_svm = classification_report(st.session_state["y_test"], y_pred_svm)
        st.text(class_report_svm)
        st.write("Confusion Matrix Support Vector Machine:")
        cm_svm = confusion_matrix(st.session_state["y_test"], y_pred_svm)
        cm_svm_df = pd.DataFrame(
            cm_svm, index=["True Class 0", "True Class 1"], columns=["Pred 0", "Pred 1"]
        )
        st.dataframe(cm_svm_df)


        st.write("3. Artificial Neural Network (ANN)")
        st.write(
            "Artificial Neural Network (ANN) adalah model machine learning yang terinspirasi dari cara kerja otak manusia. ANN terdiri dari beberapa lapisan neuron yang saling terhubung. Pada kasus ini, kita akan menggunakan ANN untuk memprediksi apakah tumor payudara bersifat benign atau malignant."
        )

        model_ann = Sequential()
        model_ann.add(
            Dense(128, input_dim=st.session_state["X_train"].shape[1], activation="relu")
        )
        model_ann.add(Dense(64, activation="relu"))
        model_ann.add(Dense(32, activation="relu"))
        model_ann.add(Dense(1, activation="sigmoid"))
        model_ann.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=5, min_lr=1e-6)
        model_ann.fit(
            st.session_state["X_train"],
            st.session_state["y_train"],
            epochs=100,
            batch_size=32,
            validation_data=(
                X_test_scaled,
                st.session_state["y_test"],
            ),  # Menggunakan data validasi
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )
        y_pred_ann = (model_ann.predict(st.session_state["X_test"]) > 0.5).astype("int32")
        accuracy_ann = accuracy_score(st.session_state["y_test"], y_pred_ann)
        st.write("Akurasi Artificial Neural Network:", accuracy_ann)
        st.write("Classification Report Artificial Neural Network:")
        class_report_ann = classification_report(st.session_state["y_test"], y_pred_ann)
        st.text(class_report_ann)
        st.write("Confusion Matrix Artificial Neural Network:")
        cm_ann = confusion_matrix(st.session_state["y_test"], y_pred_ann)
        cm_ann_df = pd.DataFrame(
            cm_ann, index=["True Class 0", "True Class 1"], columns=["Pred 0", "Pred 1"]
        )
        st.dataframe(cm_ann_df)

        # Menyimpan summary ke dalam string menggunakan StringIO
        buffer = io.StringIO()
        model_ann.summary(print_fn=lambda x: buffer.write(x + "\n"))
        summary_str = buffer.getvalue()

        # Menampilkan summary model di Streamlit dengan markdown
        st.markdown("### Model Summary:")
        st.text(summary_str)


        st.write("4. KNN")

        # sekarang buatka yang knn
        model_knn = KNeighborsClassifier(n_neighbors=5)
        model_knn.fit(st.session_state["X_train"], st.session_state["y_train"])
        y_pred_knn = model_knn.predict(st.session_state["X_test"])
        accuracy_knn = accuracy_score(st.session_state["y_test"], y_pred_knn)
        st.write("Akurasi KNN:", accuracy_knn)
        st.write("Classification Report KNN:")
        class_report_knn = classification_report(st.session_state["y_test"], y_pred_knn)
        st.text(class_report_knn)
        st.write("Confusion Matrix KNN:")
        cm_knn = confusion_matrix(st.session_state["y_test"], y_pred_knn)
        cm_knn_df = pd.DataFrame(
            cm_knn, index=["True Class 0", "True Class 1"], columns=["Pred 0", "Pred 1"]
        )
        st.dataframe(cm_knn_df)


        st.write(
            "Dari hasil evaluasi model di atas, dapat dilihat bahwa Artificial Neural Network (ANN), Logistic Regression dan SVM mmeiliki akurasi yang sama, tetapi KNN lebih rendah, Oleh karena itu ANN, LR, SVM dapat digunakan sebagai model terbaik untuk memprediksi kanker payudara."
        )

with implementasi:
    st.subheader("Implementasi Model")
st.write(
    "Pada tahap ini, kita akan menggunakan model Logistic Regression yang telah dilatih sebelumnya untuk melakukan prediksi terhadap data baru."
)

# Load model dan scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Daftar fitur yang dipilih (sesuaikan dengan fitur yang dipilih pada tahap preprocessing)
selected_features = [
    "perimeter_worst",
    "radius_worst",
    "area_worst",
	"concave points_worst",
    "concave points_mean",
    "perimeter_mean",
	"area_mean",
    "concavity_mean",
    "radius_mean",
    "area_se",
    "concavity_worst",
    "perimeter_se",
    "radius_se",
    "compactness_mean",
    "compactness_worst",
]

# Menampilkan input form untuk setiap fitur yang dipilih
st.write("Masukkan nilai untuk fitur berikut:")

input_values = {}

for feature in selected_features:
    input_values[feature] = st.number_input(
        f"Masukkan nilai untuk {feature}", min_value=0.0, step=0.1
    )

# Prediksi berdasarkan input
if st.button("Prediksi"):
    try:
        # Membuat array numpy berdasarkan input pengguna
        input_data = np.array([list(input_values.values())]).reshape(1, -1)

        # Melakukan normalisasi input data
        input_data_scaled = scaler.transform(input_data)

        # Prediksi menggunakan model
        prediction = model.predict(input_data_scaled)

        # Konversi hasil prediksi menjadi M atau B
        result = "M" if prediction == 1 else "B"

        # Menampilkan hasil prediksi
        st.write("Hasil prediksi: Tumor tersebut adalah:", result)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
