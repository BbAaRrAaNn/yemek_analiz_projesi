#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

# Dizinleri tanımla
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

RESULTS_DIR.mkdir(exist_ok=True)

def anket_verilerini_yukle():
    """Anket verilerini yükle ve temizle"""
    print("Anket verileri yükleniyor...")
    df = pd.read_csv(DATA_DIR / 'anket.csv')
    print(f"Toplam satır sayısı: {len(df)}")
    print(f"Toplam sütun sayısı: {len(df.columns)}")
    return df

def kategorileri_analiz_et(df):
    """Yemek kategorilerini analiz et"""
    print("\n=== Kategori Analizi ===")
    
    # İkinci sütun kategori bilgilerini içeriyor
    kategori_sutunu = df.columns[1]
    
    # Kategorileri say
    kategoriler = {
        'Çorbalar': 0,
        'Etli yemekler': 0,
        'Kızartmalar': 0,
        'Sebze yemekleri': 0,
        'Soslu makarnalar': 0,
        'Pilavlar': 0,
        'Tatlılar': 0,
        'Salatalar': 0
    }
    
    for idx, row in df.iterrows():
        kategori_text = str(row[kategori_sutunu])
        if pd.notna(kategori_text) and kategori_text != 'nan':
            for kategori in kategoriler.keys():
                if kategori in kategori_text or kategori.lower() in kategori_text.lower():
                    kategoriler[kategori] += 1
    
    # Sonuçları DataFrame'e dönüştür
    kategori_df = pd.DataFrame(list(kategoriler.items()), columns=['Kategori', 'Sayı'])
    kategori_df = kategori_df[kategori_df['Sayı'] > 0].sort_values('Sayı', ascending=False)
    
    print("\nKategori Dağılımı:")
    print(kategori_df.to_string(index=False))
    
    # Grafik oluştur
    plt.figure(figsize=(12, 6))
    plt.bar(kategori_df['Kategori'], kategori_df['Sayı'], color='steelblue', alpha=0.8)
    plt.xlabel('Yemek Kategorisi', fontsize=12)
    plt.ylabel('Anket Sayısı', fontsize=12)
    plt.title('Yemek Kategorilerine Göre Anket Dağılımı', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'kategori_dagilimi.png', dpi=300, bbox_inches='tight')
    print(f"Grafik kaydedildi: {RESULTS_DIR / 'kategori_dagilimi.png'}")
    plt.close()
    
    return kategori_df

def puan_analizi_yap(df):
    """Puan verilerini analiz et"""
    print("\n=== Puan Analizi ===")
    
    # Sayısal sütunları bul (puanlar)
    numeric_cols = []
    for col in df.columns[2:]:  # İlk iki sütun tarih ve kategori
        try:
            # Sütunu sayısal değere çevirmeyi dene
            pd.to_numeric(df[col], errors='coerce')
            numeric_cols.append(col)
        except:
            pass
    
    # Tüm puanların ortalamasını hesapla
    puan_listesi = []
    for col in df.columns[2:]:
        try:
            puan_serisi = pd.to_numeric(df[col], errors='coerce')
            valid_puanlar = puan_serisi.dropna()
            if len(valid_puanlar) > 0:
                puan_listesi.extend(valid_puanlar.tolist())
        except:
            pass
    
    if puan_listesi:
        puan_array = np.array(puan_listesi)
        puan_array = puan_array[(puan_array >= 1) & (puan_array <= 5)]  # 1-5 arası puanlar
        
        print(f"\nToplam puan sayısı: {len(puan_array)}")
        print(f"Ortalama puan: {np.mean(puan_array):.2f}")
        print(f"Medyan puan: {np.median(puan_array):.2f}")
        print(f"Standart sapma: {np.std(puan_array):.2f}")
        print(f"Minimum puan: {np.min(puan_array):.1f}")
        print(f"Maksimum puan: {np.max(puan_array):.1f}")
        
        # Puan dağılımı grafiği
        plt.figure(figsize=(10, 6))
        plt.hist(puan_array, bins=np.arange(0.5, 6, 0.5), color='coral', alpha=0.7, edgecolor='black')
        plt.xlabel('Puan', fontsize=12)
        plt.ylabel('Frekans', fontsize=12)
        plt.title('Genel Puan Dağılımı', fontsize=14, fontweight='bold')
        plt.xticks(np.arange(1, 6))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'puan_dagilimi.png', dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {RESULTS_DIR / 'puan_dagilimi.png'}")
        plt.close()
        
        # Puan istatistikleri
        puan_stats = {
            'Metrik': ['Ortalama', 'Medyan', 'Std Sapma', 'Minimum', 'Maksimum', 'Toplam Puan Sayısı'],
            'Değer': [
                f"{np.mean(puan_array):.2f}",
                f"{np.median(puan_array):.2f}",
                f"{np.std(puan_array):.2f}",
                f"{np.min(puan_array):.1f}",
                f"{np.max(puan_array):.1f}",
                f"{len(puan_array)}"
            ]
        }
        stats_df = pd.DataFrame(puan_stats)
        print("\nPuan İstatistikleri:")
        print(stats_df.to_string(index=False))
        
        return puan_array, stats_df
    
    return None, None

def gorsel_veri_analizi():
    """Görsel veri setini analiz et"""
    print("\n=== Görsel Veri Analizi ===")
    
    # Görsel dosyalarını say
    image_files = list(DATA_DIR.glob('*.jpeg')) + list(DATA_DIR.glob('*.jpg'))
    print(f"Toplam görsel sayısı: {len(image_files)}")
    
    # Dosya boyutlarını analiz et
    file_sizes = [f.stat().st_size / 1024 for f in image_files]  # KB cinsinden
    
    print(f"Ortalama dosya boyutu: {np.mean(file_sizes):.2f} KB")
    print(f"Minimum dosya boyutu: {np.min(file_sizes):.2f} KB")
    print(f"Maksimum dosya boyutu: {np.max(file_sizes):.2f} KB")
    
    gorsel_stats = {
        'Metrik': ['Toplam Görsel', 'Ort. Boyut (KB)', 'Min Boyut (KB)', 'Max Boyut (KB)'],
        'Değer': [
            len(image_files),
            f"{np.mean(file_sizes):.2f}",
            f"{np.min(file_sizes):.2f}",
            f"{np.max(file_sizes):.2f}"
        ]
    }
    gorsel_df = pd.DataFrame(gorsel_stats)
    print("\nGörsel İstatistikleri:")
    print(gorsel_df.to_string(index=False))
    
    return len(image_files), gorsel_df

def ozet_rapor_olustur(kategori_df, puan_array, stats_df, gorsel_count, gorsel_df):
    
    print("\n=== Özet Rapor Oluşturuluyor ===")
    
    rapor = []
    rapor.append("# YEMEK ANALİZ PROJESİ - VERİ ANALİZ RAPORU\n")
    rapor.append("=" * 60 + "\n\n")
    
    rapor.append("## 1. GENEL BİLGİLER\n")
    rapor.append(f"- Proje Adı: Üniversite Yemekhanesinde Derin Öğrenme Tabanlı Tercih Analizi\n")
    rapor.append(f"- Analiz Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    rapor.append("## 2. VERİ SETİ İSTATİSTİKLERİ\n\n")
    
    rapor.append("### 2.1 Kategori Dağılımı\n")
    rapor.append(kategori_df.to_string(index=False))
    rapor.append("\n\n")
    
    if stats_df is not None:
        rapor.append("### 2.2 Puan İstatistikleri\n")
        rapor.append(stats_df.to_string(index=False))
        rapor.append("\n\n")
    
    rapor.append("### 2.3 Görsel Veri İstatistikleri\n")
    rapor.append(gorsel_df.to_string(index=False))
    rapor.append("\n\n")
    
    rapor.append("## 3. SONUÇLAR\n\n")
    rapor.append("Veri seti başarıyla analiz edilmiştir. Toplam {} kategoride {} anket yanıtı ve {} görsel bulunmaktadır.\n".format(
        len(kategori_df), kategori_df['Sayı'].sum(), gorsel_count
    ))
    
    if puan_array is not None:
        rapor.append(f"Genel memnuniyet ortalaması: {np.mean(puan_array):.2f}/5.00\n\n")
    
    rapor.append("## 4. ÜRETİLEN ÇIKTILAR\n\n")
    rapor.append("- kategori_dagilimi.png\n")
    rapor.append("- puan_dagilimi.png\n")
    rapor.append("- veri_analiz_raporu.txt\n\n")
    
    # Raporu kaydet
    with open(RESULTS_DIR / 'veri_analiz_raporu.txt', 'w', encoding='utf-8') as f:
        f.writelines(rapor)
    
    print(f"Rapor kaydedildi: {RESULTS_DIR / 'veri_analiz_raporu.txt'}")

def main():
   
    print("=" * 60)
    print("YEMEK ANALİZ PROJESİ - VERİ ANALİZİ")
    print("=" * 60)
    
    # Anket verilerini yükle
    df = anket_verilerini_yukle()
    
    # Kategori analizi
    kategori_df = kategorileri_analiz_et(df)
    
    # Puan analizi
    puan_array, stats_df = puan_analizi_yap(df)
    
    # Görsel veri analizi
    gorsel_count, gorsel_df = gorsel_veri_analizi()
    
    # Özet rapor
    ozet_rapor_olustur(kategori_df, puan_array, stats_df, gorsel_count, gorsel_df)
    
    print("\n" + "=" * 60)
    print("VERİ ANALİZİ TAMAMLANDI!")
    print("=" * 60)

if __name__ == "__main__":
    main()

