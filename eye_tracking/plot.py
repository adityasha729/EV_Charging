import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'Local Client\eye_data.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Convert x and y values to numeric
df['x_axis'] = pd.to_numeric(df['x_axis'], errors='coerce')
df['y_axis'] = pd.to_numeric(df['y_axis'], errors='coerce')

# Drop rows with NaN in x or y
df.dropna(subset=['x_axis', 'y_axis'], inplace=True)

# Plotting
plt.figure(figsize=(12, 6.75))  # 16:9 aspect ratio (1920x1080 scaled for display)
plt.scatter(df['x_axis'], df['y_axis'], c='lime', s=10, alpha=0.7)

# Set axes to match 1920x1080 resolution
plt.xlim(0, 1920)
plt.ylim(0, 1080)
plt.gca().invert_yaxis()  # Match screen coordinates (0,0 at top-left)

# Labels and formatting
plt.title('Eye Tracking: X vs Y Coordinates (1920x1080)', fontsize=14)
plt.xlabel('X Axis (Pixels)', fontsize=12)
plt.ylabel('Y Axis (Pixels)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
