# ------------------------------
# Fire Prediction and Visualization
# ------------------------------

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import plotly.express as px
import os

# Step 2: Load CSV (auto-pick fire_data.csv)
file_path = r"C:\Users\farah\PycharmProjects\PythonProject4\fire_data.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV file not found at {file_path}")

df = pd.read_csv(file_path)

# Step 3: Quick look at data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Step 4: Preprocess
df['date'] = pd.to_datetime(df['year_month'])  # use year_month column
df = df.rename(columns={"fire_area_percent": "fire_area_percentage"})  # unify column name
df = df.sort_values('date')
df = df[['date', 'fire_intensity', 'fire_area_percentage']]

# Step 5: Historical Visualization
plt.figure(figsize=(12,5))
plt.plot(df['date'], df['fire_intensity'], label='Fire Intensity')
plt.plot(df['date'], df['fire_area_percentage'], label='Fire Area %')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Historical Fire Trends')
plt.legend()
plt.show()

sns.lineplot(x='date', y='fire_intensity', data=df, label='Fire Intensity')
sns.lineplot(x='date', y='fire_area_percentage', data=df, label='Fire Area %')
plt.show()

# Step 6: Forecast Fire Area Percentage with Prophet
df_area = df[['date', 'fire_area_percentage']].rename(columns={'date':'ds', 'fire_area_percentage':'y'})
model_area = Prophet(yearly_seasonality=True, daily_seasonality=False)
model_area.fit(df_area)

future_area = model_area.make_future_dataframe(periods=24, freq='M')
forecast_area = model_area.predict(future_area)

model_area.plot(forecast_area)
plt.title('Fire Area % Forecast')
plt.show()

model_area.plot_components(forecast_area)
plt.show()

# Step 7: Forecast Fire Intensity with Prophet
df_intensity = df[['date', 'fire_intensity']].rename(columns={'date':'ds', 'fire_intensity':'y'})
model_intensity = Prophet(yearly_seasonality=True, daily_seasonality=False)
model_intensity.fit(df_intensity)

future_intensity = model_intensity.make_future_dataframe(periods=24, freq='M')
forecast_intensity = model_intensity.predict(future_intensity)

model_intensity.plot(forecast_intensity)
plt.title('Fire Intensity Forecast')
plt.show()

model_intensity.plot_components(forecast_intensity)
plt.show()

# Step 8: Interactive Plot (Optional)
fig = px.line(forecast_area, x='ds', y='yhat', title='Predicted Fire Area % (Interactive)')
fig.show()

fig2 = px.line(forecast_intensity, x='ds', y='yhat', title='Predicted Fire Intensity (Interactive)')
fig2.show()

# Step 9: Save Predictions to CSV
predictions = pd.DataFrame({
    'date': forecast_area['ds'],
    'predicted_fire_area_percentage': forecast_area['yhat'],
    'predicted_fire_intensity': forecast_intensity['yhat']
})

output_file = r"C:\Users\farah\PycharmProjects\PythonProject4\fire_predictions.csv"
predictions.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

# ------------------------------
# Script End
# ------------------------------
