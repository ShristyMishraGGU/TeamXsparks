import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- USER CONFIG ----------
VIDEO_PATH = r"C:\Users\farah\Downloads\vid1.mp4"
OUTPUT_CSV = r"C:\Users\farah\PycharmProjects\PythonProject4\predictions.csv"
OUTPUT_PLOT = r"C:\Users\farah\PycharmProjects\PythonProject4\temp_trend.png"

start_year = 2010
frames_per_year = 20
sample_rate = 1
calib_min_temp = 0.0
calib_max_temp = 50.0
hot_temp_threshold = 35.0
poly_degree = 2
target_year = 2030

# ---------- Helper functions ----------
def frame_to_temperature_map(frame):
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    denom = (r + g + b) + 1e-6
    heat = (r - b) / denom
    heat = np.clip(heat, 0.0, 1.0)
    temp = calib_min_temp + heat * (calib_max_temp - calib_min_temp)
    return temp

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    temps, hot_percents = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            temp_map = frame_to_temperature_map(frame)
            mean_temp = float(np.nanmean(temp_map))
            hot_pct = float((temp_map > hot_temp_threshold).sum()) / (temp_map.size) * 100.0
            temps.append(mean_temp)
            hot_percents.append(hot_pct)
        idx += 1
    cap.release()
    print(f'Read {idx} frames, sampled {len(temps)} frames.')
    return temps, hot_percents

def aggregate_to_years(temps, hot_percents, start_year, frames_per_year):
    years, mean_temps, mean_hot = [], [], []
    n = len(temps)
    group_count = int(np.ceil(n / frames_per_year))
    for g in range(group_count):
        s = g * frames_per_year
        e = min((g + 1) * frames_per_year, n)
        yrs = temps[s:e]
        hps = hot_percents[s:e]
        years.append(start_year + g)
        mean_temps.append(float(np.mean(yrs)))
        mean_hot.append(float(np.mean(hps)))
    return np.array(years), np.array(mean_temps), np.array(mean_hot)

def forecast_poly(years, values, year_target_max, degree=2):
    coeffs = np.polyfit(years, values, degree)
    p = np.poly1d(coeffs)
    years_out = np.arange(years[0], year_target_max + 1)
    preds = p(years_out)
    return years_out, preds

# ---------- Main ----------
if __name__ == '__main__':
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found at {VIDEO_PATH}.")

    temps, hot_percents = process_video(VIDEO_PATH)
    years_hist, mean_temps_hist, mean_hot_hist = aggregate_to_years(temps, hot_percents, start_year, frames_per_year)

    years_out, preds_temp = forecast_poly(years_hist, mean_temps_hist, target_year, degree=poly_degree)
    _, preds_hot = forecast_poly(years_hist, mean_hot_hist, target_year, degree=1)

    # Save CSV
    df = pd.DataFrame({'year': years_out})
    hist_map_temp = dict(zip(years_hist, mean_temps_hist))
    hist_map_hot = dict(zip(years_hist, mean_hot_hist))
    df['mean_temp'] = [hist_map_temp.get(int(y), np.nan) for y in df['year']]
    df['hot_area_pct'] = [hist_map_hot.get(int(y), np.nan) for y in df['year']]
    df['pred_mean_temp'] = preds_temp
    df['pred_hot_area_pct'] = preds_hot
    df.to_csv(OUTPUT_CSV, index=False)
    print('Saved CSV with historical + predicted values to', OUTPUT_CSV)

    # Save static plot
    plt.figure(figsize=(8, 5))
    plt.plot(years_hist, mean_temps_hist, marker='o', label='Historical mean temp')
    plt.plot(years_out, preds_temp, marker='x', linestyle='--', label=f'Predicted (poly deg={poly_degree})')
    plt.xlabel('Year')
    plt.ylabel('Mean temperature (C)')
    plt.title('Mean Temperature: historical + forecast')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print('Saved static trend plot to', OUTPUT_PLOT)
