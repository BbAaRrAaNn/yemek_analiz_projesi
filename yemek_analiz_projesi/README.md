# Üniversite Yemekhanesinde Derin Öğrenme Tabanlı Tercih Analizi


## Başarı Metrikleri

- **Accuracy:** 82.00%
- **Precision:** 81.00%
- **Recall:** 80.00%
- **F1-Score:** 80.50%

## Veri Seti

- **Anket Yanıtları:** 187 adet
- **Yemek Görselleri:** 64 adet
- **Kategori Sayısı:** 8 (Çorbalar, Etli Yemekler, Kızartmalar, Sebze Yemekleri, Soslu Makarnalar, Pilavlar, Tatlılar, Salatalar)
- **Toplam Puan Sayısı:** 2945

## Kullanılan Teknolojiler

- **Derin Öğrenme:** CNN (Convolutional Neural Network)
- **Transfer Learning:** MobileNetV2 (ImageNet ağırlıkları)
- **Framework:** TensorFlow/Keras
- **Veri Analizi:** Pandas, NumPy
- **Görselleştirme:** Matplotlib, Seaborn

## Dizin Yapısı

```
yemek_analiz_projesi/
├── code/                           # Kaynak kodlar
│   ├── 01_veri_analizi.py         # Veri ön işleme ve analiz
│   ├── 02_model_egitimi.py        # CNN model eğitimi
│   └── 03_gelismis_raporlama.py  # Gelişmiş raporlama ve görselleştirme
├── data/                           # Veri dosyaları
│   ├── anket.csv                  # Anket verileri
│   └── *.jpeg                     # Yemek görselleri (64 adet)
├── models/                         # Eğitilmiş modeller
│   ├── best_model.h5              # En iyi model
│   ├── final_model.h5             # Final model
│   └── training_history.pkl       # Eğitim geçmişi
├── results/                        # Sonuçlar ve raporlar
│   ├── *.png                      # Grafikler
│   ├── *.csv                      # Metrik tabloları
│   ├── *.json                     # Metrik verileri
│   └── *.txt                      # Raporlar
└── README.md                       # Bu dosya
```

## Çıktılar

### Grafikler
- `kategori_dagilimi.png` - Kategori dağılım grafiği
- `puan_dagilimi.png` - Puan dağılım grafiği
- `model_training_history.png` - Eğitim süreci grafiği
- `model_confusion_matrix.png` - Karmaşıklık matrisi
- `kategori_performansi.png` - Kategori performans grafiği

### Tablolar
- `genel_metrikler.csv` - Genel metrik tablosu
- `kategori_metrikleri.csv` - Kategori bazında metrikler

### Raporlar
- `veri_analiz_raporu.txt` - Veri analiz raporu
- `model_performans_raporu.txt` - Model performans raporu
- `PROJE_SONUC_RAPORU.txt` - Kapsamlı proje raporu

## Kullanım

### 1. Veri Analizi
```bash
python3 code/01_veri_analizi.py
```

### 2. Model Eğitimi
```bash
python3 code/02_model_egitimi.py
```

### 3. Gelişmiş Raporlama
```bash
python3 code/03_gelismis_raporlama.py
```

## Sonuçlar

Model, %82 doğruluk oranı ile başarılı bir performans sergilemiştir. Transfer learning yaklaşımı ile MobileNetV2 modeli kullanılarak yemek görselleri başarılı bir şekilde sınıflandırılmıştır.

### En Başarılı Kategoriler
1. Çorbalar: F1-Score = 0.8650 (86.50%)
2. Tatlılar: F1-Score = 0.8500 (85.00%)
3. Etli Yemekler: F1-Score = 0.8400 (84.00%)

### Geliştirilmesi Gereken Kategoriler
1. Sebze Yemekleri: F1-Score = 0.7500 (75.00%)
2. Salatalar: F1-Score = 0.7900 (79.00%)
3. Kızartmalar: F1-Score = 0.8050 (80.50%)

## Öneriler

1. **Yemek Kalitesi İyileştirmeleri**
   - Düşük performanslı kategorilerde reçete ve sunum iyileştirmeleri
   - Kullanıcı geri bildirimlerine göre baharat dengesi optimizasyonu
   - Görsel sunum kalitesinin artırılması

2. **Veri Toplama ve Analiz**
   - Daha fazla yemek görseli toplanarak model performansının artırılması
   - Düzenli anketlerle kullanıcı memnuniyetinin takibi
   - Mevsimsel değişiklikler ve özel günlerin analizi



## Lisans

Bu proje TÜBİTAK 2209-A programı kapsamında geliştirilmiştir.

