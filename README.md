# Uganda Grain Price Analysis and Forecasting 
 
This project analyzes Ugandan grain commodity prices and produces 12 month forecasts for 2026 using time series and ML models. It supports district and regional views and produces outputs for Power BI. 
 
## Scope 
- Commodities: Maize, Sorghum White, Sorghum Red, Beans Yellow, Beans Nambaale, Beans Wairimu, Barley in highland districts, Soya 
- Geography: Central, Western, Northern, Eastern Uganda across 24 districts plus 3 barley only highland districts Kigezi, Kapchorwa, Kabale 
- Data: Daily retail prices in UGX per kg for Jan 2024 to Dec 2025 
- Forecast horizon: Jan 2026 to Dec 2026, 12 months 
- Sources used for calibration: FEWS NET, WFP VAM, Agahikaine Grains, FAO GIEWS 
 
## Methods in the notebook 
Section outline from Uganda_Grain_SARIMA_Analysis.ipynb: 
1. Setup and data loading 
2. Exploratory data analysis 
3. Seasonal analysis 
4. Stationarity tests and ACF and PACF 
5. SARIMA modeling and forecasting with a custom implementation 
6. SARIMAX with exogenous regional demand 
7. Forecast visualization 
8. Diagnostics and validation with backtesting 
9. Performance summary 
10. Final summary and forecast table 
11. Machine learning and deep learning models including Random Forest, SVR, XGBoost, Prophet, and LSTM 
12. Price change analysis for MoM, QoQ, and YoY 
13. Appendix with model references and data sources 
 
## Data and outputs 
- CSV datasets: PBI_Uganda_Grains_Daily.csv, PBI_Uganda_Grains_Monthly.csv, PBI_Uganda_Grains_Forecasts.csv, PBI_Model_Statistics.csv 
- Excel files: Uganda_Grain_Prices_Daily_2024_2025.xlsx, PowerBI_Dashboard_Guide.xlsx 
- Figures: outputs folder contains fig1_overview through fig19_price_change_summary and a backtest heatmap 
 
## Data generation and maintenance scripts 
- rebuild_crops.py rebuilds the core CSVs using district multipliers and seasonal profiles 
- rebuild_excel_crops.py rebuilds the Excel deliverables 
- add_soya.py adds the soya series to CSVs, Excel, and the notebook 
- patch_ and fix_ scripts update or repair notebook cells and charts 
 
## How to use 
1. Open Uganda_Grain_SARIMA_Analysis.ipynb in Jupyter and run the cells in order. 
2. Regenerate data with rebuild_crops.py or add_soya.py if you need a fresh dataset. 
3. Use the CSV and PNG outputs in Power BI or reports.
