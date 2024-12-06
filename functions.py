import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def penjelasan_fitur(feature):
    fitur = {
        "diagnosis": 'Diagnosis kanker, dengan kategori "M" untuk malignan dan "B" untuk benign.',
        "radius_mean": "Rata-rata dari jarak antara pusat dan perimeter tumor.",
        "texture_mean": "Rata-rata tekstur permukaan tumor berdasarkan analisis fraktal.",
        "perimeter_mean": "Rata-rata perimeter atau keliling tumor.",
        "area_mean": "Rata-rata luas tumor.",
        "smoothness_mean": "Rata-rata kelancaran permukaan tumor.",
        "compactness_mean": "Rata-rata kompaktnya tumor, dihitung dari perimeter dan area.",
        "concavity_mean": "Rata-rata kekonkavitas tumor.",
        "concave points_mean": "Rata-rata jumlah titik cekung pada tumor.",
        "symmetry_mean": "Rata-rata simetri tumor.",
        "fractal_dimension_mean": "Rata-rata dimensi fraktal permukaan tumor.",
        "radius_se": "Standard error dari radius tumor.",
        "texture_se": "Standard error dari tekstur tumor.",
        "perimeter_se": "Standard error dari perimeter tumor.",
        "area_se": "Standard error dari area tumor.",
        "smoothness_se": "Standard error dari kelancaran permukaan tumor.",
        "compactness_se": "Standard error dari kompaktnya tumor.",
        "concavity_se": "Standard error dari kekonkavitas tumor.",
        "concave points_se": "Standard error dari jumlah titik cekung tumor.",
        "symmetry_se": "Standard error dari simetri tumor.",
        "fractal_dimension_se": "Standard error dari dimensi fraktal permukaan tumor.",
        "radius_worst": "Nilai terburuk dari radius tumor.",
        "texture_worst": "Nilai terburuk dari tekstur tumor.",
        "perimeter_worst": "Nilai terburuk dari perimeter tumor.",
        "area_worst": "Nilai terburuk dari area tumor.",
        "smoothness_worst": "Nilai terburuk dari kelancaran permukaan tumor.",
        "compactness_worst": "Nilai terburuk dari kompaktnya tumor.",
        "concavity_worst": "Nilai terburuk dari kekonkavitas tumor.",
        "concave points_worst": "Nilai terburuk dari jumlah titik cekung tumor.",
        "symmetry_worst": "Nilai terburuk dari simetri tumor.",
        "fractal_dimension_worst": "Nilai terburuk dari dimensi fraktal permukaan tumor.",
    }
    return fitur.get(feature, "Penjelasan tidak tersedia.")


def plot_distribusi_kelas(df):
    col1, col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(6, 4))
        ax = df["diagnosis"].value_counts().plot(kind="pie", autopct="%1.1f%%")
        ax.set_ylabel("")
        ax.legend(title="Kelas", loc="center left", bbox_to_anchor=(1, 0.5))
        st.pyplot(plt.gcf())

    with col2:
        st.subheader("Jumlah Data pada Setiap Kelas")
        st.write(df["diagnosis"].value_counts())


def plot_distribusi_fitur_kelas(data, target_column, bins=30):
    # Pisahkan data berdasarkan kelas target
    data_benign = data[data[target_column] == 0]  # Kelas Benign (0)
    data_malignant = data[data[target_column] == 1]  # Kelas Malignant (1)

    # Tentukan fitur-fitur yang akan diplot (kecuali kolom target)
    feature_columns = [col for col in data.columns if col != target_column]

    # Plot distribusi tiap fitur dengan target
    plt.figure(figsize=(15, 20))
    for i, feature in enumerate(feature_columns, 1):
        if feature == "id" or feature == "Unnamed: 32":
            continue
        plt.subplot((len(feature_columns) + 4) // 5, 5, i)
        plt.hist(data_benign[feature], bins=bins, alpha=0.6, label="Benign (0)")
        plt.hist(data_malignant[feature], bins=bins, alpha=0.6, label="Malignant (1)")
        plt.title(feature)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())  # Menampilkan plot di Streamlit


def menampilkan_informasi_data(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Statistik Deskriptif")
        st.table(df.describe())
    with col2:
        st.write("Tipe Data Fitur")
        st.table(df.dtypes)
    with col3:
        st.write("Pengecekan Missing Value")
        st.table(df.isnull().sum())


def menampilkan_outlier(df):
    # Membuat dua kolom untuk layout yang lebih rapi di Streamlit
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Pengecekan Outlier")
        st.write("Dilakukan dengan menggunakan metode IQR (Interquartile Range).")

        # Menghitung Q1, Q3, dan IQR untuk setiap kolom numerik
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Membuat DataFrame untuk menampilkan Q1, Q3, IQR, dan batas atas/bawah
        outlier_table = pd.DataFrame(
            {
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
            }
        )

        # Menampilkan tabel yang menunjukkan batas IQR
        st.table(outlier_table)

        # Menampilkan informasi bahwa data yang berada di luar batas dianggap sebagai outlier
        st.write("Data yang berada di luar batas atas dan bawah adalah outlier.")

    with col2:
        st.write("### Jumlah Oulier pada setiap fitur")
        outliers = (df < lower_bound) | (df > upper_bound)
        outlier_counts = outliers.sum()
        st.table(outlier_counts)
        st.write(
            "Karrena terdapat outler maka nanti pada tahap preprocessing akan dilakukan handling outlier., abaikan kolom id karena nanti akan kita drop."
        )


def tampilkan_heatmap_korelasi(data):
    # Menghapus kolom 'id' dan 'Unnamed: 32' sebelum menghitung korelasi
    correlation_matrix = data.corr()

    # Membuat figure untuk heatmap
    plt.figure(figsize=(20, 16))

    # Menggambar heatmap dengan seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

    # Menambahkan judul pada grafik
    plt.title("Matrix Korelasi")

    # Menampilkan heatmap di Streamlit
    st.pyplot(plt.gcf())  # Menggunakan Streamlit untuk menampilkan plot


def handle_outliers(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Capping
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])

    return data


def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return Q1, Q3, IQR, lower_bound, upper_bound


def penjelasan_IQR():
    np.random.seed(10)
    data = np.random.normal(0, 1, 100)  # Data normal (mean=0, std=1)
    data_with_outliers = np.append(data, [10, 12, -9, -11])  # Menambahkan outlier

    # Menampilkan penjelasan tentang IQR
    st.subheader("Penjelasan tentang IQR dan Deteksi Outlier ")
    st.write(
        """
    **Interquartile Range (IQR)** adalah rentang antara **Q1 (first quartile)** dan **Q3 (third quartile)**, yang menunjukkan sebaran 50% data tengah.
    - **Q1** adalah nilai yang memisahkan 25% data terendah.
    - **Q3** adalah nilai yang memisahkan 75% data terendah.
    - **IQR** dihitung dengan mengurangi Q1 dari Q3: **IQR = Q3 - Q1**.

    Outliers adalah nilai yang berada di luar batas normal yang ditentukan berdasarkan IQR:
    - **Lower Bound** = Q1 - 1.5 * IQR
    - **Upper Bound** = Q3 + 1.5 * IQR

    Data yang berada di luar batas ini dianggap sebagai **outliers**.
    """
    )

    Q1, Q3, IQR, lower_bound, upper_bound = detect_outliers(data_with_outliers)
    # Menampilkan batas IQR dan informasi outlier
    st.write(f"Q1 (First Quartile): {Q1:.2f}")
    st.write(f"Q3 (Third Quartile): {Q3:.2f}")
    st.write(f"IQR (Interquartile Range): {IQR:.2f}")
    st.write(f"Lower Bound (Batas Bawah): {lower_bound:.2f}")
    st.write(f"Upper Bound (Batas Atas): {upper_bound:.2f}")
    outliers = (data_with_outliers < lower_bound) | (data_with_outliers > upper_bound)
    outliers_count = np.sum(outliers)
    # Menampilkan jumlah outliers
    st.write(f"Jumlah Outliers: {outliers_count} dari {len(data_with_outliers)} data")
    # Visualisasi Boxplot untuk melihat distribusi data dan outliers
    st.write("### Visualisasi Boxplot untuk Deteksi Outlier")
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_with_outliers, vert=False)
    plt.title("Boxplot - Deteksi Outlier Menggunakan IQR")
    plt.xlabel("Nilai")
    st.pyplot(plt)
    # Menampilkan data yang dianggap outliers
    outlier_values = data_with_outliers[outliers]
    st.write("### Nilai-nilai Outliers yang Ditemukan:")
    st.write(outlier_values)
