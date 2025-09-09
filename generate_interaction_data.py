import csv
import random

# Provided kiosk data (each row is a kiosk, not a station)
kiosk_rows = [
    # KioskID, Action, SelectedImage, SelectedAttack, MobileUPI, ChargeAmount, PointsChange, ChargeCost, CyberAttackLoss, StartingCharge, EndingCharge
    [3, "start_charging", "deepfake4.png", "Cyber Attack", "", 25, 0, 50, 10, 22, 47],
    [1, "start_charging", "real6.png", "Safe", "", 60, 0, 60, 0, 27, 87],
    [1, "start_charging", "real6.png", "Cyber Attack", "", 20, 0, 40, 10, 51, 71],
    [3, "start_charging", "deepfake1.png", "Cyber Attack", "", 15, 0, 30, 10, 52, 67],
    [2, "start_charging", "real6.png", "Cyber Attack", "", 10, 0, 20, 10, 46, 56],
    [1, "start_charging", "real4.png", "Safe", "", 20, 0, 20, 0, 30, 50],
    [2, "start_charging", "deepfake5.png", "Safe", "", 20, 0, 20, 0, 28, 48],
    [2, "start_charging", "real6.png", "Cyber Attack", "", 10, 0, 20, 10, 26, 36],
    [1, "start_charging", "real4.png", "Safe", "", 30, 0, 30, 0, 17, 47],
    [3, "start_charging", "real4.png", "Safe", "", 20, 0, 20, 0, 21, 41],
    [3, "start_charging", "deepfake4.png", "Cyber Attack", "", 7, 0, 15, 10, 32, 39],
    [2, "start_charging", "real6.png", "Safe", "", 20, 0, 20, 0, 23, 43],
    [3, "start_charging", "deepfake1.png", "Cyber Attack", "", 12, 0, 25, 10, 17, 29],
    [1, "start_charging", "deepfake3.png", "Safe", "", 20, 0, 20, 0, 13, 33],
    [1, "start_charging", "real6.png", "Safe", "", 40, 0, 40, 0, 4, 44],
    [3, "start_charging", "deepfake3.png", "Cyber Attack", "", 10, 0, 20, 10, 27, 37],
    [1, "start_charging", "real4.png", "Safe", "", 25, 0, 25, 0, 19, 44],
    [2, "start_charging", "real4.png", "Safe", "", 25, 0, 25, 0, 31, 56],
    [3, "start_charging", "real4.png", "Safe", "", 20, 0, 20, 0, 40, 60],
    [2, "start_charging", "deepfake2.png", "Cyber Attack", "", 7, 0, 15, 10, 45, 52],
]

header = [
    'StationID', 'Kiosk1_SelectedImage', 'Kiosk1_SelectedAttack', 'Kiosk1_MobileUPI', 'Kiosk1_ChargeAmount',
    'Kiosk1_PointsChange', 'Kiosk1_ChargeCost', 'Kiosk1_CyberAttackLoss', 'Kiosk1_StartingCharge', 'Kiosk1_EndingCharge',
    'Kiosk2_SelectedImage', 'Kiosk2_SelectedAttack', 'Kiosk2_MobileUPI', 'Kiosk2_ChargeAmount',
    'Kiosk2_PointsChange', 'Kiosk2_ChargeCost', 'Kiosk2_CyberAttackLoss', 'Kiosk2_StartingCharge', 'Kiosk2_EndingCharge',
    'Kiosk3_SelectedImage', 'Kiosk3_SelectedAttack', 'Kiosk3_MobileUPI', 'Kiosk3_ChargeAmount',
    'Kiosk3_PointsChange', 'Kiosk3_ChargeCost', 'Kiosk3_CyberAttackLoss', 'Kiosk3_StartingCharge', 'Kiosk3_EndingCharge'
]

# Create 20 stations, each with 3 kiosks (cycle through provided kiosks)
stations = []
for i in range(20):
    station_id = i + 1
    kiosks = []
    for k in range(3):
        idx = (i * 3 + k) % len(kiosk_rows)
        kiosk = kiosk_rows[idx]
        # Only use kiosk data columns (skip kiosk_id and action)
        kiosks.append(kiosk[2:])
    row = [station_id]
    for kiosk in kiosks:
        row.extend(kiosk)
    stations.append(row)

# Write initial 20 stations to CSV
with open('interaction_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for row in stations:
        writer.writerow(row)

# Duplicate to 400 rows (randomize cyber attack for kiosks with no attack)
data = stations.copy()
while len(data) < 400:
    row = random.choice(stations)
    row_copy = row.copy()
    for k in range(3):
        idx_attack = 1 + k*9 + 1  # attack column index for kiosk k+1
        if row_copy[idx_attack] == '':
            if random.random() < 0.3:
                row_copy[idx_attack] = 'Cyber Attack'
    data.append(row_copy)

with open('interaction_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)

print("interaction_data.csv generated with 400 rows.")
