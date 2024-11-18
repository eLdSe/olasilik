import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

api_key = "key"   # API anahtarı
url = "https://api.api-ninjas.com/v1/cars"
headers = {
    "X-Api-Key": api_key
}

def get_car_info_by_make(make):
    params = {
        "make": make
    }
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if not data:
            print(f"{make} markası için veri yok.")
        else:
            return data
    else:
        print(f"Veri talebi sırasında hata oluştu: {response.status_code}")
        return None

def plot_fuel_consumption(make):
    cars = get_car_info_by_make(make)
    if cars:
        models = [car['model'] for car in cars]
        city_mpg = [car['city_mpg'] for car in cars]
        highway_mpg = [car['highway_mpg'] for car in cars]

        x = range(len(models))

        plt.figure(figsize=(10, 6))
        plt.bar(x, city_mpg, width=0.4, label='Şehir içi MPG', align='center', color='blue')
        plt.bar(x, highway_mpg, width=0.4, label='Otoyol MPG', align='edge', color='green')

        plt.xlabel('Araç Modelleri')
        plt.ylabel('Yakıt Tüketimi (MPG)')
        plt.title(f'{make.capitalize()} Modelleri için Yakıt Tüketimi Karşılaştırması')
        plt.xticks(x, models, rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()

def display_car_info(make):
    cars = get_car_info_by_make(make)
    if cars:
        print(f"{make.capitalize()} markası için araç bilgileri:")
        for car in cars:
            print("\nAraç Bilgileri:")
            print(f"Marka: {car.get('make', 'Bilinmiyor')}")
            print(f"Model: {car.get('model', 'Bilinmiyor')}")
            print(f"Yıl: {car.get('year', 'Bilinmiyor')}")
            print(f"Yakıt Tipi: {car.get('fuel_type', 'Bilinmiyor')}")
            print(f"Şehir içi yakıt tüketimi (MPG): {car.get('city_mpg', 'Bilinmiyor')}")
            print(f"Otoyol yakıt tüketimi (MPG): {car.get('highway_mpg', 'Bilinmiyor')}")
            print(f"Şanzıman Tipi: {car.get('transmission', 'Bilinmiyor')}")
            print(f"Silindir Sayısı: {car.get('cylinders', 'Bilinmiyor')}")
            print(f"Motor (L): {car.get('displacement', 'Bilinmiyor')}")
            print(f"Çekiş Tipi: {car.get('drive', 'Bilinmiyor')}")
            print("-" * 40)

def plot_fuel_comparison_for_brands(brands):
    brand_names = []
    avg_city_mpg = []
    avg_highway_mpg = []
    
    for brand in brands:
        cars = get_car_info_by_make(brand)
        if cars:
            city_mpg = [car['city_mpg'] for car in cars]
            highway_mpg = [car['highway_mpg'] for car in cars]
            
            avg_city_mpg.append(sum(city_mpg) / len(city_mpg))
            avg_highway_mpg.append(sum(highway_mpg) / len(highway_mpg))
            brand_names.append(brand.capitalize())

    if brand_names:
        x = range(len(brand_names))
        plt.figure(figsize=(10, 6))
        plt.bar(x, avg_city_mpg, width=0.4, label='Şehir içi MPG', align='center', color='blue')
        plt.bar(x, avg_highway_mpg, width=0.4, label='Otoyol MPG', align='edge', color='green')

        plt.xlabel('Markalar')
        plt.ylabel('Ortalama Yakıt Tüketimi (MPG)')
        plt.title('Farklı Markalar İçin Yakıt Tüketimi Karşılaştırması')
        plt.xticks(x, brand_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Belirtilen markalar için veri alınamadı.")

def regression_analysis(make):
    cars = get_car_info_by_make(make)
    if cars:
        # Şehir içi ve otoyol MPG verilerini çıkarıyoruz
        city_mpg = [car['city_mpg'] for car in cars if car['city_mpg'] is not None]
        highway_mpg = [car['highway_mpg'] for car in cars if car['highway_mpg'] is not None]

        if len(city_mpg) > 1 and len(highway_mpg) > 1:
            # Regresyon için verilerin hazırlanması
            X = np.array(city_mpg).reshape(-1, 1)  # Bağımsız değişken (city_mpg)
            y = np.array(highway_mpg)  # Bağımlı değişken (highway_mpg)

            # Lineer regresyon modeli oluşturuluyor
            model = LinearRegression()
            model.fit(X, y)

            # Regresyon katsayılarının yazdırılması
            print(f"Regresyon katsayısı (eğim): {model.coef_[0]:.3f}")
            print(f"Serbest terim (intercept): {model.intercept_:.3f}")

            # Grafik çizimi
            plt.figure(figsize=(8, 5))
            plt.scatter(city_mpg, highway_mpg, color='blue', label='Veriler')
            plt.plot(city_mpg, model.predict(X), color='red', label='Regresyon Hattı')
            plt.xlabel('Şehir içi MPG')
            plt.ylabel('Otoyol MPG')
            plt.title(f"{make.capitalize()} Markası İçin Regresyon Analizi")
            plt.legend()
            plt.show()
        else:
            print(f"{make} markası için yeterli veri yok.")
    else:
        print(f"{make} markası için veri alınamadı.")

# İki marka için hipotez testi
def hypothesis_testing(make1, make2):
    # İki marka için verileri alıyoruz
    cars1 = get_car_info_by_make(make1)
    cars2 = get_car_info_by_make(make2)
    
    if cars1 is None or cars2 is None:
        print(f"Bir veya her iki marka için veri alınamadı: {make1}, {make2}")
        return
    
    # Her iki marka için şehir içi yakıt tüketimi (city_mpg) verilerini çıkarıyoruz
    city_mpg1 = [car['city_mpg'] for car in cars1 if car['city_mpg'] is not None]
    city_mpg2 = [car['city_mpg'] for car in cars2 if car['city_mpg'] is not None]

    if not city_mpg1 or not city_mpg2:
        print(f"{make1} ve/veya {make2} markası için yeterli veri yok.")
        return
    
    # Bağımsız örnekler için t-testini yapıyoruz
    t_stat, p_value = stats.ttest_ind(city_mpg1, city_mpg2)
    
    # Test sonuçlarını yazdırıyoruz
    print(f"{make1.capitalize()} ve {make2.capitalize()} Markaları İçin t-Test Sonuçları:")
    print(f"t-istatistiği: {t_stat:.3f}")
    print(f"p-değeri: {p_value:.3f}")
    
    # Sonuçları histogram ile görselleştiriyoruz
    plt.figure(figsize=(10, 6))
    
    # İlk marka için histogram
    plt.hist(city_mpg1, bins=10, alpha=0.5, label=f"{make1.capitalize()} Şehir İçi MPG", color='blue')
    # İkinci marka için histogram
    plt.hist(city_mpg2, bins=10, alpha=0.5, label=f"{make2.capitalize()} Şehir İçi MPG", color='green')
    
    plt.xlabel("Şehir İçi MPG")
    plt.ylabel("Frekans")
    plt.title(f"{make1.capitalize()} ve {make2.capitalize()} Markaları İçin Yakıt Tüketimi Karşılaştırması")
    plt.legend(loc='upper right')
    plt.show()

def calculate_and_plot_average_fuel_consumption(make):
    cars = get_car_info_by_make(make)
    if not cars:
        print(f"{make.capitalize()} markası için veri yok.")
        return None
    
    total_city_mpg = sum(car['city_mpg'] for car in cars if car['city_mpg'] is not None)
    total_highway_mpg = sum(car['highway_mpg'] for car in cars if car['highway_mpg'] is not None)
    count = len(cars)
    
    avg_city_mpg = total_city_mpg / count
    avg_highway_mpg = total_highway_mpg / count

    print(f"{make.capitalize()} markası için ortalama yakıt tüketimi (MPG):")
    print(f"Şehir içi MPG: {avg_city_mpg:.2f}")
    print(f"Otoyol MPG: {avg_highway_mpg:.2f}")
    
    # Grafik çizimi
    plt.figure(figsize=(8, 5))
    labels = ['Şehir içi MPG', 'Otoyol MPG']
    mpg_values = [avg_city_mpg, avg_highway_mpg]
    plt.bar(labels, mpg_values, color=['blue', 'green'])
    plt.xlabel('Yakıt Tüketimi Türü')
    plt.ylabel('Ortalama MPG')
    plt.title(f"{make.capitalize()} Markası İçin Ortalama Yakıt Tüketimi")
    plt.show()

# Ana Menü
def menu():
    print("\nAraç Bilgileri Analiz Programına Hoş Geldiniz")
    print("1. Araçlar hakkında bilgi al")
    print("2. Yakıt Tüketimi Karşılaştırması Grafiği")
    print("3. Farklı Markalar İçin Yakıt Tüketimi Karşılaştırması")
    print("4. Regresyon Analizi (Şehir içi ve Otoyol MPG)")
    print("5. İki Marka Arasında Hipotez Testi")
    print("6. Ortalama Yakıt Tüketimini Hesapla ve Göster")
    print("0. Çıkış")
    
    choice = input("Seçiminizi yapın (0-6): ")
    
    if choice == '1':
        make = input("Araç markasını girin: ")
        display_car_info(make)
    elif choice == '2':
        make = input("Araç markasını girin: ")
        plot_fuel_consumption(make)
    elif choice == '3':
        brands = input("Karşılaştırmak istediğiniz markaları virgülle ayırarak girin: ").split(",")
        brands = [brand.strip() for brand in brands]
        plot_fuel_comparison_for_brands(brands)
    elif choice == '4':
        make = input("Araç markasını girin: ")
        regression_analysis(make)
    elif choice == '5':
        make1 = input("İlk marka adını girin: ")
        make2 = input("İkinci marka adını girin: ")
        hypothesis_testing(make1, make2)
    elif choice == '6':
        make = input("Araç markasını girin: ")
        calculate_and_plot_average_fuel_consumption(make)
    elif choice == '0':
        print("Çıkılıyor...")
        exit()
    else:
        print("Geçersiz seçenek. Lütfen tekrar deneyin.")


if __name__ == "__main__":
    while True:
        menu()
