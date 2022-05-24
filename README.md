# Machine Learning Regression

Membuat Machine Learning untuk mengelola dan membuat pemodelan dari data yang sudah diberikan.
Data yang akan digunakan adalah data Daegu Apartemen 

sumber data : [Data Daegu Apartment](https://drive.google.com/file/d/1MPDotXZNmiq6geRi8BkGd-fjttzoyTxW/view)

### **Contents**

1. Business Problem Understanding
2. Data Understanding
3. Data Preprocessing
4. Modeling
5. Conclusion
6. Recommendation


### **Business Problem Understanding**
**Context**

Apartemen adalah sebuah hunian yang banyak diminati oleh orang-orang yang tinggal di perkotaan, apalagi di kota-kota besar atau kota metropolitan yang padat populasi seperti Kota Daegu di Korea Selatan, merujuk dari laman wikipedia mengenai kota [Daegu](https://id.wikipedia.org/wiki/Daegu) adalah kota metropolitan terbesar nomor empat di negara Korea Selatan.
apartemen sendiri dipilih oleh orang-orang yang tinggal di daerah perkotaan tersebut adalah dikarenakan untuk membangun rumah tapak atau membeli rumah tapak perlu mengeluarkan biaya yang sangat tinggi, sehingga apartemen adalah hunian yang dipilih orang-orang yang tinggal di perkotaan atau kota besar. Selain itu, apartemen sendiri menyediakan berbagai fasilitas, seperti kolam renang, gym, minimarket/convenience store, keamanan, dan fasilitas yang lainnya, sehingga dengan harga yang sama untuk membeli rumah pun, ada kecenderungan orang-orang lebih memilih apartemen karna fasilitas yang diberikan oleh pihak pengelola apartemen itu sendiri.
Banyak agen properti yang menawarkan unit apartemen untuk dipasarkan atau dijual kepada orang-orang yang tinggal di area perkotaan yang padat. Untuk menjadi agen properti apartemen yang dapat menjual harga properti yang ideal, caranya sangat mudah, hanya dengan membuat iklan atau penawaran bisa dengan memasang iklan pada halaman website agen properti tersebut apabila memiliki atau pada digital platform untuk forum jual beli properti dan mencantumkan harga properti yang akan dijual. Lalu, agen properti apartemen tersebut bisa mulai memasukkan daftar properti apartemennya beserta harga jual yang sudah ditentukan oleh agen tersebut. Tetapi untuk memasarkan properti berupa unit apartemen, agen properti harus merujuk kepada harga yang tepat supaya mendapat keuntungan yang bagus atau dalam artian tidak terlalu murah tetapi sesuai dengan harga yang seharusnya dikeluarkan oleh pembeli agar para calon pembeli mau untuk membeli unit yang sedang dipasarkan.


**Problem Statement**
Salah satu tantangan terbesar bagi agen properti apartemen adalah pemecahan masalah untuk dapat memiliki model bisnis yang menguntungkan secara finansial bagi pemilik properti/agen, serta dapat memberikan pengalaman positif terhadap pembeli properti apartemen.
Mengingat bahwa harga penawaran harus ideal, untuk menentukan harga properti mereka, dengan hanya memberikan petunjuk minimal yang memungkinkan agen properti untuk menawarkan unit properti apartemennya dengan membandingkan tempat serupa di lingkungan mereka untuk mendapatkan harga yang kompetitif. Agen properti pun dapat memasukkan harga yang lebih tinggi untuk fasilitas tambahan apa pun yang mereka anggap perlu. **Dengan bertambahnya agen properti yang ada, untuk menentukan harga yang tepat untuk dapat tetap kompetitif di lingkungan sekitar unit apartemen yang akan ditawarkan sangatlah penting**.


**Goals**
Berdasarkan permasalahan tersebut, agen properti harus atau sudah memiliki 'tool' yang dapat memprediksi serta membantu mereka mereka (dalam hal ini agen properti) untuk dapat **menentukan harga jual yang tepat untuk tiap unit properti yang baru akan mereka jual**. Adanya perbedaan pada berbagai fasilitas yang terdapat pada suatu properti apartemen yang akan dijual, seperti tipe propertinya, lokasi, fasilitas dapat menambah keakuratan prediksi harga jual, yang mana dapat mendatangkan profit bagi agen properti, dan juga tentunya masih terjangkau bagi pembeli.


**Analytic Approach**
Jadi, yang perlu saya lakukan adalah menganalisis data untuk dapat menemukan pola dari fitur-fitur yang ada, yang membedakan properti apartemen dengan yang lainnya. 
Selanjutnya, saya akan membangun suatu model regresi yang akan membantu agen properti untuk dapat menyediakan 'tool' prediksi harga jual yang ideal, yang mana akan berguna untuk agen properti dalam menentukan harga jual listing-nya.


**Metric Evaluation**
Evaluasi metrik yang akan digunakan adalah RMSE, MAE, dan MAPE, di mana RMSE adalah nilai rataan akar kuadrat dari error, MAE adalah rataan nilai absolut dari error, sedangkan MAPE adalah rataan persentase error yang dihasilkan oleh model regresi. Semakin kecil nilai RMSE, MAE, dan MAPE yang dihasilkan, berarti model semakin akurat dalam memprediksi harga jual sesuai dengan limitasi fitur yang digunakan. 
Selain itu, kita juga bisa menggunakan nilai R-squared atau adj. R-squared jika model yang nanti terpilih sebagai final model adalah model linear. Nilai R-squared digunakan untuk mengetahui seberapa baik model dapat merepresentasikan varians keseluruhan data. Semakin mendekati 1, maka semakin fit pula modelnya terhadap data observasi. Namun, metrik ini tidak valid untuk model non-linear.



## **Data Understanding**
#### Attributes Information
| **Attribute** | **Data Type** | **Description** |
| --- | --- | --- |
| HallwayType | Object | Tipe Lorong |
| TimeToSubway | Object | Jarak Tempuh Ke Stasiun Kereta Bawah Tanah |
| SubwayStation| Object | Nama Stasiun Bawah Tanah |
| N_FacilitiesNearBy(ETC) | Float | Jumlah Fasilitas Terdekat Dari Apartemen |
| N_FacilitiesNearBy(PublicOffice) | Float | Jumlah Fasilitas Umum Terdekat Dari Apartemen |
| N_SchoolNearBy(University) | Float | Jumlah Sekolah Terdekat Dari Apartemen |
| N_Parkinglot(Basement) | Float | Jumlah Ketersediaan Parkir (Basement) |
| YearBuilt | Integer | Tahun Pembuatan |
| N_FacilitiesInApt |Integer | Jumlah Fasilitas Yang Berada Di Apartemen |
| Size(sqf) | Integer | Ukuran Luas Apartemen (square feet/kaki persegi) |
| SalePrice | Integer | Harga Jual |

<br>

### **EDA**
Dilakukan sedikit EDA (Exploratory Data Analysist) untuk melihat lebih detail dari unique fitur yang ada beserta tipe data nya sebagai salah satu metode saya melakukan feature selection.
Berdasarkan informasi data yang diperoleh, disekitar subway station Kyungbuk Uni Hospital adalah lokasi apartemen terbanyak , diikuti oleh Myung-duk, dari dataset yang diberikan, masing - masing berjumlah :
- Kyungbuk Uni Hospital : 1152
- Myung-duk : 1044

dimana nilai nya bisa mencapa 2x (dua kali) lipat dari jumlah masing-masing apartemen yang berada disekitar subway station lainnya.
untuk jenis tipe lorong apartemennya paling banyak adalah terraced dan mixed. Masing-masing bernilai :
- Terraced : 2528
- Mixed : 1131
- Corridor : 464


#### **Data Preprocessing**
Pada tahap ini, saya akan melakukan cleaning pada data yang nantinya data yang sudah dibersihkan akan saya gunakan untuk proses analisis selanjutnya. Beberapa hal yang perlu dilakukan adalah:
- Drop fitur yang tidak memiliki relevansi terhadap permasalahan yang sedang dihadapi.
- Melakukan treatment terhadap missing value jika ada. Bisa dengan cara men-drop fiturnya jika memang tidak dibutuhkan atau bisa juga dengan mengimputasi dengan nilai yang paling masuk akal berdasarkan kasusnya.
Untuk proses data preprocessing dan feature engineering, kita akan menggunakan dataframe hasil duplikasi dari dataframe yang sebelumnya digunakan.

**Cek Missing Value**
dari informasi data yang didapat diatas, tidak terdapat adanya missing value sehingga tidak ada treatment kusus yang harus diberikan atau drop column yang harus dilakukan.

**Cek Duplikasi Data**
terdapat duplikasi data sebanyak 1422, sehingga kita harus melakukan penanganan yaitu dengan melakukan drop duplikasi data.

**Seleksi Fitur**
Tentu perlu ada pertimbangan sebelum melakukan drop pada kolom atau fitur. Sebagai pertimbangan, pada tahap ini, menggunakan domain knowledge untuk memutuskan kira-kira fitur mana saja yang dirasa tidak memiliki relevansi.
- Jika ditinjau berdasar domain knowledge, semua fitur yang ada dalam dataset tersebut saling berelevansi satu sama lain, sehingga tidak perlukan adanya drop kolom.
- dan dari jumlah feature yang terbatas, sehingga saya tidak melakukan drop columns

**Data Correlation**
Cek Korelasi SubwayStation, TimeToSubway, HallwayType terhadap SalePrice,
untuk tipe data object pada dataset ini, akan dilakukan pengecekan korelasi menggunakan Association Correlation supaya tidak perlu mengganti tipe data dari fitur tersebut.
Association Correlation digunakan untuk melihat hubungan antar kolom dimana kolom tersebut bertipe object

Dari hasil korelasi asosiasi & korelasi matrix menunjukan tidak ada fitur yang korelasi tinggi, tetapi dapat dilihat bahwa ada beberapa fitur yang memiliki korelasi positif dan negaitf, yaitu :

Korelasi Positif (+) :
- Size(sqf)
- HallwayType
- N_FacilitiesInApt       
- TimeToSubway            

Korelasi Negatif (-)
- N_FacilitiesNearBy(PublicOffice)
- N_FacilitiesNearBy(ETC)
- N_SchoolNearBy(University)

Korelasi matrix digunakan untuk melihat hubungan antar kolom yang bertipe numerik, disini saya menggunakan pearson correlation dikarenakan saya ingin melihat linear correlation dimana pearson correlation sangat cocok digunakan untuk mencari hubungan linear correlation.

##**Modeling**
**splitting Data**
saya melakukan splitting data dengan proporsi train 80% dan test 20%

----------------------------------------------------------------------------------------
saya menggunakan def function Eva_Matrix_DF untuk melakukan pengecekan evaluation matrix

**Pipeline**
Saya menggunakan pipeline untuk menghindari data leakage (kebocoran data).
Yang dimaksud data Leakage atau kebocoran data adalah kondisi dimana model sudah melihat data test, padahal seharusnya model hanya boleh melihat data train.
Kesimpulan dari PipeLine adalah agar data tidak menghafal dari hasil data train sebelum-sebelumnya

**Encode Data**
saya mengelempokan feature berdasarkan tipe datanya, yaitu :
- numerikal 
- kategorikal 
alasannya adalah karena saya akan menerapkan metode feature engineering yang berbeda untuk ke dua kelompok feature tersebut.

**penjelasan** 
Saya membuat dua jenis pipeline untuk feature engineering, yaitu:
- `numeric_pipeline`
- `categoric_pipeline`
didalam numeric_pipeline, saya melakukan feature engineering yaitu `PolynomialFeatures` dengan degree 3 dan tidak mengikutkan    
bias nya (pangkat 0). 
Selain PolynomialFeatures saya juga melakukan PowerTransformer dengan menggunakan metode `yeo-johnson`.
`numeric_pipeline` ini saya gunakan untuk melakukan feature engineering pada kelompok feature numeric columns
Didalam `categoric_pipeline`, saya melakukan feature engineering yaitu `OneHotEncoder`, saya menggunakan one hot encoder untuk melakukan encoding kolom bertipe data kategorikal yang bersifat nominal.
`categoric_pipeline` ini saya gunakan untuk melakukan feature engineering pada kelompok feature categoric columns

**penjelasan**
Saya melakukan tambahan feature engineering yaitu 
- scalling
- non scalling
menggunakan 2 metode feature engineering scaling untuk algoritma yang berbasis jarak. contohnya seperti knn svm dll, untuk yang non scalling saya gunakan untuk algoritma yang berbasis pohon, contohnya : XGB, randomg forest dll.
selanjutnya kita akan melakukan step untuk training data

### Hasil dari Evaluation Matrix :
Dapat dilihat bahwa dari hasil Evaluation Matrix diatas menghasilkan nilai yaitu :
- 2 Model dengan Nilai RMSE terbaik adalah Model Random Forest dan XGBoost dengan hasil yang berbeda sangat tipis, RF = 47923.23 dan XGB = 47918.23
- 2 Model tersebut memiliki nilai RMSE yang lebih rendah dibandingkan dengan Model lain, sehingga perlu dilakukan Hyper Parameter Tuning
- Model dengan Nilai terburuk adalah Model SVR dengan hasil nilai RMSE yang tinggi
- Model yang akan dilakukan Tunning adalah XGB

### **Improvement / Tuning**
##### XG Boost Hyper Parameter Tuning
**Penjelasan**
alasan saya improvement menggunakan XGB karena hasil evaluation matrixnya yang terbaik atau paling bagus diantara algoritma yang lainnya seperti :
- Random Forest
- SVR
- Decision Tree
sehingga saya melakukan tuning hyperparameternya menggunakan XGB
- `XGB` sendiri, atau memiliki nama lain Extreme Gradient Boosting adalah salah satu algoritme paling populer yang berbasis tree. Dimana XGB sendiri adalah versi upgrade dari algoritma Gradient Boosting milik `SKLEARN`.

**cara kerja XGB**
- cara kerjanya sendiri menyerupai seperti kita bermain golf, dimana `hole` adalah target utama
- kemudian model akan secara pelan-pelan melakukan perbaikan (pukulan) sampai mencapai target (bola masuk hole)

**Hasil Akhir**
**Before Tuning & After Tuning**
Dapat dilihat bahwa dari Model yang sudah di Tunning terdapat perbedaan Nilai pada RMSE
- Nilai RMSE Testing XGB awal 47918.23 turun menjadi 47682.54

**Analisis Error untuk model terbaik**
**Penjelasan**

dari sampel error, dimana selisih antara data aktual dan data prediksi untuk SalePrice.
berarti menunjukan bahwa model bisa saja prediksinya meleset kurang atau lebih dari harga aktual dengan rata-rata error sebesar **38890.38** (dari nilai MAE nya)
Dari hasil error yang ditampilkan, nilai error terbesar dari hasil model tuning XGB yaitu sebesar : 126395.34, itu menandakan bahwa :
- Prediksi dari SalePrice bisa saja OverEstimation terbesarnya yaitu 126395.34
- Prediksi dari SalePrice bisa saja UnderEstimation terbesarnya 126395.34
Adanya nilai-nilai error yang besar membuat perbedaan yang cukup signifikan antara nilai RMSE dan MAE. Hal ini dapat tergambarkan pula pada plot di atas, di mana terdapat harga aktual yang rendah tapi diprediksi jauh lebih tinggi (overestimation), dan juga sebaliknya (underestimation). Akan tetapi, nilai RMSE yang didapat, yaitu sebesar 47682.54 menjadikan model ini dapat dikategorikan ke dalam 'reasonable forecasting'.
Dari hasil yang didapatkan, error terbesar disebabkan dari HallwayType : Terraced

## **Conlusion**
Berdasarkan pemodelan yang sudah dilakukan, fitur 'Size' dan 'N_FacilitiesApt' menjadi fitur yang paling berpengaruh terhadap 'SalePrice'.
Metrik evaluasi yang digunakan pada model adalah nilai R2, MAE, MSE & RMSE. Jika ditinjau dari nilai RMSE yang dihasilkan oleh model setelah dilakukan hyperparameter tuning, yaitu sebesar 47682.54, kita dapat menyimpulkan bahwa bila nanti model yang kita buat ini digunakan untuk memperkirakan harga SalePrice baru di Daegu pada rentang nilai seperti yang dilatih terhadap model (maksimal harga 585840.00) **untuk satuan harga tidak diketahui, karena tidak ada di deskripsi data atau dictionary data yang diberikan**, maka perkiraan harganya rata-rata akan meleset kurang lebih sebesar **38890.38** dari harga yang mungkin seharusnya. Tetapi, tidak menutup kemungkinan juga prediksinya meleset karena bias yang dihasilkan model masih cukup tinggi bila dilihat dari visualisasi antara harga aktual dan prediksi. Bias yang dihasilkan oleh model ini dikarenakan oleh fitur HallwayType yaitu Terraced dan kurangnya fitur pada dataset yang bisa merepresentasikan aspek properti apartemen dan juga services seperti jumlah kamar dalam unit apartemen, biaya maintenance dan lain-lain.

##**Recomendation**
Hal-hal yang dapat dilakukan untuk mengembangkan model agar lebih baik lagi, seperti:
- Mengecek prediksi mana saja yang memiliki nilai error yang tinggi. Kita dapat mengelompokkan error tersebut ke dalam grup overestimation dan underestimation, lalu memilih 5% error paling ekstrim saja untuk tiap grup. Nantinya pengelompokkan akan menjadi 3 grup, yaitu overestimation (5%), underestimation (5%), dan grup mayoritas yang error-nya mendekati nilai mean (90%). Setelahnya kita bisa mengecek hubungan antara error tersebut dengan tiap variabel independen. Pada akhirnya kita dapat mengetahui sebenarnya variabel mana saja dan aspek apa yang menyebabkan model menghasilkan error yang tinggi, sehingga kita bisa melakukan training ulang dengan penerapan feature engineering lainnya.
- Jika memungkinkan, penambahan fitur yang lebih korelatif dengan target ('SalePrice'), seperti jumlah kamar dalam satu unit,  jarak ke pusat kota & biaya maintenance apartemen. Selain itu, adanya penambahan data terkini untuk Daegu Apartemen Data tentu akan dapat mengimprovisasi kapasitas prediksi dari model. Namun, kalau jumlah data dan fiturnya masih seperti dataset ini, kemungkinan besar tidak akan mengubah hasilnya secara signifikan.
