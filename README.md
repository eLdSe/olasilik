# 🚗 Car Fuel Consumption Analysis

A Python-based data analysis project that retrieves car data from an external API and performs fuel consumption analysis, visualization, and statistical testing.

This project demonstrates working with APIs, data processing, visualization, regression analysis, and hypothesis testing using popular Python libraries.

---

## 🚀 Project Overview

The application fetches car information by brand using the API Ninjas Cars API and allows users to analyze fuel consumption data (MPG) in multiple ways.  
It provides interactive visualizations, statistical comparisons, and regression analysis through a console-based menu system.

The project is designed as an educational data analysis project and showcases practical usage of data science and Python analytics tools.

---

## 🧩 Features

- Fetch car data by brand using a REST API  
- Display detailed car specifications  
- Compare city vs highway fuel consumption (MPG)  
- Compare average fuel consumption across multiple brands  
- Visualize data using bar charts and histograms  
- Perform linear regression analysis (city MPG vs highway MPG)  
- Conduct hypothesis testing (t-test) between two car brands  
- Interactive console menu for easy navigation  

---

## 📊 Analysis & Visualizations

The project includes:
- Fuel consumption comparison charts  
- Regression line visualization  
- Histogram-based comparison for hypothesis testing  
- Average MPG calculations and plots  

All visualizations are generated using `matplotlib` and `seaborn`.

---

## 🛠 Technologies Used

- Python  
- requests  
- numpy  
- pandas  
- matplotlib  
- seaborn  
- scipy  
- scikit-learn  

---

⚙️ Installation & Run

Clone the repository:

git clone https://github.com/your-username/car-fuel-consumption-analysis.git


Go to the project directory:

cd car-fuel-consumption-analysis


Install required dependencies:

pip install -r requirements.txt


Run the program:

python main.py

---

## 🔑 API Usage

This project uses the **API Ninjas Cars API**.

You must provide your own API key:

```python
api_key = "YOUR_API_KEY"


