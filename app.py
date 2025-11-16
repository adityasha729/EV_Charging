from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify, flash
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import random
import csv
import os
from datetime import datetime, timezone
import base64
from vision_utils import generate_caption_gemini
from rag import RAGStore
import threading
import time
from datetime import datetime as _dt
from eyetrax import GazeEstimator
import cv2
import json
import os
import numpy as np



# ==================== CONFIGURATION PARAMETERS ====================
# All configurable parameters are centralized here for easy modification

class KioskThresholds:
    GREEN = 0.3  # ≤ this is green
    YELLOW = 0.7  # > green and ≤ this is yellow, > this is red

class GameConfig:
    """Configuration class containing all game parameters"""
    
    # File paths
    DEEPFAKE_FOLDER = 'images/deepfake'
    REAL_FOLDER = 'images/real'
    SETUP_CSV = 'setup.csv'
    DEMOGRAPHICS_CSV = 'demographics.csv'
    USER_INTERACTIONS_CSV = 'user_interactions.csv'
    INTERACTION_DATA_CSV = 'interactions_rag/interaction_data.csv'
    #EYE_DATA_CSV = 'eye_cam_tracking_rag/eye_data.csv'
    #INTERACTION_DATA_CSV = 'interactions_rag_eye/interaction_data.csv'
    EYE_DATA_CSV = 'eye_cam_tracking_rag_eye/eye_data.csv'
    HELP_REQUESTS_CSV = 'help_requests.csv'
    EEG_DATA_CSV = 'eeg_data.csv'
    
    
    # Default game parameters (will be overridden by setup.csv if available)
    DEFAULT_STARTING_POINTS = 1000
    DEFAULT_INITIAL_CHARGE_MIN = 20
    DEFAULT_INITIAL_CHARGE_MAX = 40
    DEFAULT_NUMBER_OF_TRIALS = 10
    DEFAULT_COST_PER_MINUTE = 10
    DEFAULT_CHARGE_RATE_PER_MINUTE = 5
    DEFAULT_DRIVING_DECREMENT_MIN = 5
    DEFAULT_DRIVING_DECREMENT_MAX = 15
    DEFAULT_IMAGE_SELECTION_THRESHOLD = 0.5
    DEFAULT_DEEPFAKE_ATTACK_PROBABILITY = 0.7
    DEFAULT_REAL_ATTACK_PROBABILITY = 0.3
    DEFAULT_DISCOUNT_COUPON_CHANCE = 0.2
    DEFAULT_DISCOUNT_COUPON_PERCENTAGE = 0.15
    DEFAULT_CYBER_ATTACK_CHARGE_REDUCTION = 0.5
    DEFAULT_CYBER_ATTACK_LOSS_POINTS = 50
    DEFAULT_COUPON_CODE_IMAGE_PROB = 0.3
    
    # Game logic parameters
    MAX_CHARGE = 100
    MIN_CHARGE = 0
    MIN_POINTS = 0
    
    # File extensions
    ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']

class SimulationConfig:
    RAG = True  # Whether to use RAG for chatbot context
    RAG_TOP_K = 3  # Number of similar descriptions to retrieve for RAG
    ScreenshotProcessing = False  # Whether a screenshot is being processed
    GazeTracking = True  # Whether gaze tracking is enabled




# ==================== FLASK APP INITIALIZATION ====================


def get_participant_id():
    return session.get('participant_id', 'unknown')

def get_screenshot_history_path():
    participant_id = get_participant_id()
    os.makedirs('station_screenshots', exist_ok=True)
    return f'station_screenshots/screenshot_history_{participant_id}.json'

def get_charging_context_path():
    participant_id = get_participant_id()
    os.makedirs('station_screenshots', exist_ok=True)
    return f'station_screenshots/charging_context_{participant_id}.json'

def load_json_file(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def save_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def get_screenshot_context():
    # Return the latest screenshot description for this participant
    path = get_screenshot_history_path()
    history = load_json_file(path, {})
    if not history:
        return None
    # Return the latest (highest trial) description
    try:
        latest_trial = max(map(int, history.keys()))
        return history[str(latest_trial)]
    except Exception:
        return None

def set_screenshot_context(description):
    # Store the latest screenshot description for this participant
    # (for compatibility, but not used for history)
    pass


load_dotenv()
app = Flask(__name__)
app.secret_key = 'a_very_secret_random_string_1234567890!@#$%^&*()'

# Chatbot configuration

# Load model and API keys from environment variables

chatbots = {
    "gemini": {
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model": os.environ.get("GEMINI_MODEL", "")
    },
    "openai": {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "")
    }
}

# ==================== UTILITY FUNCTIONS ====================

def ensure_directories_exist():
    """Create necessary directories if they don't exist"""
    os.makedirs(GameConfig.DEEPFAKE_FOLDER, exist_ok=True)
    os.makedirs(GameConfig.REAL_FOLDER, exist_ok=True)
    os.makedirs('eye_data_with_marker', exist_ok=True)
    os.makedirs('eye_cam_tracking', exist_ok=True)


def clear_eye_data_file():
    """Clear/initialize the eye data CSV with header on each app start."""
    path = GameConfig.EYE_DATA_CSV
    # ensure parent directory exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Timestamp (ISO 8601)', 'X', 'Y', 'Page'])


# Global small shared state for gaze recorder to know the current page
gaze_state = {
    'page': 'unknown'
}
gaze_state_lock = threading.Lock()

gaze_context = {
    'context': []
}
gaze_context_lock = threading.Lock()


def _gaze_recorder_loop(output_path, camera_index=0, model_path="webcam_eye/gaze_model.pkl", fps=30):
    """Background gaze recorder loop (simplified from webcam_eye/gaze_recorder.record).
    Writes rows to `output_path` with columns: Timestamp, Timestamp (ISO 8601), X, Y, Page
    Does not record blink data.
    """
    try:
        if GazeEstimator is None or cv2 is None:
            print("Gaze recorder: eyetrax or cv2 not available; recorder not started.")
            return
        estimator = GazeEstimator()
        try:
            estimator.load_model(model_path)
        except Exception:
            # model loading may fail if model not present; continue using estimator if possible
            pass

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Gaze recorder: cannot open camera {camera_index}")
            return

        # Try to set a frame rate (may be ignored)
        try:
            cap.set(cv2.CAP_PROP_FPS, fps)
        except Exception:
            pass

        # Open CSV and append rows
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # If file is empty, write header
            if f.tell() == 0:
                writer.writerow(['Timestamp', 'Timestamp (ISO 8601)', 'X', 'Y', 'Page'])

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # extract features and predict gaze; tolerate failures
                x = y = None
                try:
                    features, blink = estimator.extract_features(frame)
                    if features is not None:
                        try:
                            pred = estimator.predict([features])[0]
                            x, y = pred
                        except Exception:
                            x = y = None
                except Exception:
                    x = y = None

                # clamp/validate coordinates
                try:
                    screen_w = int(1920)
                    screen_h = int(1080)
                    buffer = int(100)
                except Exception:
                    screen_w, screen_h, buffer = 1920, 1080, 100

                if x is not None and y is not None:
                    if (x < -buffer) or (x > screen_w + buffer) or (y < -buffer) or (y > screen_h + buffer):
                        x = None
                        y = None
                    else:
                        x = max(0, min(int(x), screen_w))
                        y = max(0, min(int(y), screen_h))

                ts = time.time()
                iso = _dt.now().isoformat()
                with gaze_state_lock:
                    page = gaze_state['page']
                    if page == 'unknown':
                        continue  # skip writing unknown page samples
                    if page[:5] == "index":
                        with gaze_context_lock:
                            if x is not None and y is not None:
                                gaze_context['context'].append([x,y])
                                
                

                # Only write sample rows when both x and y are valid (user requested)
                if x is not None and y is not None:
                    writer.writerow([f"{ts:.6f}", iso, int(x), int(y), page])
                f.flush()
                frame_idx += 1

    except Exception as e:
        print(f"Gaze recorder loop failed: {e}")


def start_gaze_recorder_once():
    """Start the gaze recorder background thread (idempotent)."""
    if getattr(start_gaze_recorder_once, 'started', False):
        return
    start_gaze_recorder_once.started = True
    t = threading.Thread(target=_gaze_recorder_loop, args=(GameConfig.EYE_DATA_CSV,), kwargs={'camera_index': 0, 'model_path': 'webcam_eye/gaze_model.pkl', 'fps': 30}, daemon=True)
    t.start()

def load_images():
    """Load available images from directories"""
    deepfake_images = [f for f in os.listdir(GameConfig.DEEPFAKE_FOLDER) 
                      if any(f.lower().endswith(ext) for ext in GameConfig.ALLOWED_IMAGE_EXTENSIONS)]
    real_images = [f for f in os.listdir(GameConfig.REAL_FOLDER) 
                  if any(f.lower().endswith(ext) for ext in GameConfig.ALLOWED_IMAGE_EXTENSIONS)]
    return deepfake_images, real_images

def load_setup_variables():
    """Load game parameters from setup.csv or use defaults"""
    setup_variables = {
        'starting_points': GameConfig.DEFAULT_STARTING_POINTS,
        'initial_charge_min': GameConfig.DEFAULT_INITIAL_CHARGE_MIN,
        'initial_charge_max': GameConfig.DEFAULT_INITIAL_CHARGE_MAX,
        'number_of_trials': GameConfig.DEFAULT_NUMBER_OF_TRIALS,
        'cost_per_minute': GameConfig.DEFAULT_COST_PER_MINUTE,
        'charge_rate_per_minute': GameConfig.DEFAULT_CHARGE_RATE_PER_MINUTE,
        'driving_decrement_min': GameConfig.DEFAULT_DRIVING_DECREMENT_MIN,
        'driving_decrement_max': GameConfig.DEFAULT_DRIVING_DECREMENT_MAX,
        'image_selection_threshold': GameConfig.DEFAULT_IMAGE_SELECTION_THRESHOLD,
        'deepfake_attack_probability': GameConfig.DEFAULT_DEEPFAKE_ATTACK_PROBABILITY,
        'real_attack_probability': GameConfig.DEFAULT_REAL_ATTACK_PROBABILITY,
        'discount_coupon_chance': GameConfig.DEFAULT_DISCOUNT_COUPON_CHANCE,
        'discount_coupon_percentage': GameConfig.DEFAULT_DISCOUNT_COUPON_PERCENTAGE,
        'cyber_attack_charge_reduction': GameConfig.DEFAULT_CYBER_ATTACK_CHARGE_REDUCTION,
        'cyber_attack_loss_points': GameConfig.DEFAULT_CYBER_ATTACK_LOSS_POINTS,
        'coupon_code_image_prob': GameConfig.DEFAULT_COUPON_CODE_IMAGE_PROB
    }
    
    try:
        with open(GameConfig.SETUP_CSV, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    try:
                        setup_variables[row[0]] = float(row[1])
                    except ValueError:
                        print(f"Warning: Could not parse setup value for {row[0]}: {row[1]}")
    except FileNotFoundError:
        print(f"Warning: {GameConfig.SETUP_CSV} not found. Using default values.")
    
    return setup_variables

def initialize_csv_files():
    """Initialize CSV files with headers if they don't exist"""
    
    # Demographics CSV
    if not os.path.exists(GameConfig.DEMOGRAPHICS_CSV):
        with open(GameConfig.DEMOGRAPHICS_CSV, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Participant ID', 'Timestamp', 'Age', 'Gender', 'Education', 
                           'Specialization', 'EV Experience'])
    
    # User interactions CSV
    if not os.path.exists(GameConfig.USER_INTERACTIONS_CSV):
        with open(GameConfig.USER_INTERACTIONS_CSV, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Participant ID', 'Timestamp', 'Event', 'Trial', 'Station ID', 'Action', 
                'Selected Image Shown', 'Selected Attack', 'Mobile/UPI', 'Charge Amount', 
                'Points Change', 'Charge Cost', 'Cyber Attack Loss', 'Starting Charge', 
                'Ending Charge', 'Station 1 Status', 'Station 1 Image', 'Station 1 Attack', 
                'Station 1 Probabilities', 'Station 1 Random Numbers', 'Station 2 Status', 
                'Station 2 Image', 'Station 2 Attack', 'Station 2 Probabilities', 
                'Station 2 Random Numbers', 'Station 3 Status', 'Station 3 Image', 
                'Station 3 Attack', 'Station 3 Probabilities', 'Station 3 Random Numbers', 
                'Game Start Time', 'Game End Time', 'Charge Time', 'Drive Time', 
                'Decision Time', 'Random Numbers',
                'Chatbot Prompt',           
                'Chatbot Suggested Kiosk'  
            ])
    
    # Eye data CSV
    if not os.path.exists(GameConfig.EYE_DATA_CSV):
        with open(GameConfig.EYE_DATA_CSV, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Timestamp (ISO 8601)', 'X', 'Y', 'Page'])
    
    # Help requests CSV
    if not os.path.exists(GameConfig.HELP_REQUESTS_CSV):
        with open(GameConfig.HELP_REQUESTS_CSV, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Participant ID', 'Timestamp', 'Help Text', 'Trial Number'])
    
    # Interaction data CSV
    if not os.path.exists(GameConfig.INTERACTION_DATA_CSV):
        with open(GameConfig.INTERACTION_DATA_CSV, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'StationID', 'Kiosk1_SelectedImage', 'Kiosk1_SelectedAttack', 'Kiosk1_MobileUPI', 'Kiosk1_ChargeAmount', 
                'Kiosk1_PointsChange', 'Kiosk1_ChargeCost', 'Kiosk1_CyberAttackLoss', 'Kiosk1_StartingCharge', 'Kiosk1_EndingCharge',
                'Kiosk2_SelectedImage', 'Kiosk2_SelectedAttack', 'Kiosk2_MobileUPI', 'Kiosk2_ChargeAmount', 
                'Kiosk2_PointsChange', 'Kiosk2_ChargeCost', 'Kiosk2_CyberAttackLoss', 'Kiosk2_StartingCharge', 'Kiosk2_EndingCharge',
                'Kiosk3_SelectedImage', 'Kiosk3_SelectedAttack', 'Kiosk3_MobileUPI', 'Kiosk3_ChargeAmount', 
                'Kiosk3_PointsChange', 'Kiosk3_ChargeCost', 'Kiosk3_CyberAttackLoss', 'Kiosk3_StartingCharge', 'Kiosk3_EndingCharge', 'Chatbot_Response'
            ])
    
    # Mouse data CSV
    if not os.path.exists('mouse_data.csv'):
        with open('mouse_data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Participant ID', 'Timestamp', 'X', 'Y', 'Page', 'Event'])

def is_game_over():
    """Check if game should end based on charge and trial count"""
    trial = session.get('trial', 0)
    charge = session.get('charge', 0)
    points = session.get('points', 0)
    
    # Game ends if: exceeded max trials OR (no charge AND no points to buy charge)
    max_trials_reached = trial > setup_variables['number_of_trials']
    no_charge_and_no_points = charge <= 0 and points <= 0
    
    return max_trials_reached or no_charge_and_no_points

def can_afford_charging(charge_amount):
    """Check if player can afford to charge for given amount"""
    charge_cost = charge_amount * setup_variables['cost_per_minute']
    return session.get('points', 0) >= charge_cost

def generate_charging_stations():
    """Generate charging station data for current trial"""
    # Randomly decide whether there will be 0 or 1 occupied stations
    occupied_stations = random.sample([1, 2, 3], k=random.choice([0, 1]))
    
    charging_stations = []
    for i in range(1, 4):
        station = {
            'id': i,
            'name': f'Kiosk {i}',
            # 'location': f'Location {i}',
            'status': 'Occupied' if i in occupied_stations else 'Available',
            'image': None,
            'attack': None,
            'probabilities': {},
            'random_numbers': {},
            'discount_coupon': False
        }
        
        if station['status'] == 'Available':
            # Generate random numbers for this station
            image_selection = random.random()
            attack_determination = random.random()
            discount_coupon_chance = random.random()
            
            # Determine image and attack
            if image_selection < setup_variables['image_selection_threshold'] and deepfake_images:
                station['image'] = random.choice(deepfake_images)
                prob_attack = setup_variables['deepfake_attack_probability']
                station['attack'] = 'Cyber Attack' if attack_determination < prob_attack else 'Safe'
                station['probabilities']['deepfake'] = prob_attack
            elif real_images:
                station['image'] = random.choice(real_images)
                prob_attack = setup_variables['real_attack_probability']
                station['attack'] = 'Cyber Attack' if attack_determination < prob_attack else 'Safe'
                station['probabilities']['real'] = prob_attack
            else:
                station['image'] = 'default_image.png'
                station['attack'] = 'Safe'
            
            # Determine discount coupon availability
            station['discount_coupon'] = discount_coupon_chance < setup_variables['discount_coupon_chance']
            
            # Store random numbers for logging
            station['random_numbers'] = {
                'image_selection': image_selection,
                'attack_determination': attack_determination,
                'discount_coupon_chance': discount_coupon_chance
            }
        
        charging_stations.append(station)
    
    return charging_stations

# ==================== DATA LOGGING FUNCTIONS ====================

def send_eeg_marker(marker):
    """Send marker to EEG system and update EEG data"""
    try:
        session['eeg_marker'] = marker
        print(f"Sent marker to EEG: {marker}")
        update_eeg_data_with_marker(marker)
    except Exception as e:
        print(f"EEG marker request failed: {e}")

def send_eye_marker(marker):
    """Send marker to eye tracking system and update eye data"""
    try:
        session['eye_marker'] = marker
        print(f"Sent marker to Eye: {marker}")
        # Also update the eye-cam page label used by the gaze recorder
        try:
            send_eye_cam_marker(marker)
        except Exception:
            # defensive: continue even if updating cam label fails
            pass
        update_eye_data_with_marker(marker)
    except Exception as e:
        print(f"Eye marker request failed: {e}")


def send_eye_cam_marker(marker):
    """Set the gaze recorder's page label directly to `marker`.

    This simplified function stores the marker in session (if available) and writes
    the marker string into `gaze_state['page']` under a lock. The recorder will tag
    subsequent samples with this value.
    """
    try:
        try:
            session['eye_cam_marker'] = marker
        except Exception:
            pass

        page_label = marker if marker is not None else 'Error_Unknown'
        page_label = str(page_label).lstrip('/')
        if page_label == '':
            page_label = 'root'

        with gaze_state_lock:
            gaze_state['page'] = page_label
            if page_label[:5] == "index":
                with gaze_context_lock:
                    gaze_context['context'] = []  # reset context for new index page
            print(f"Updated gaze_state page -> {gaze_state['page']}")
        print(page_label)
    except Exception as e:
        print(f"send_eye_cam_marker failed: {e}")

def update_eeg_data_with_marker(marker):
    """Update EEG data file with marker"""
    participant_id = session.get('participant_id', 'unknown')
    input_filename = GameConfig.EEG_DATA_CSV
    output_filename = f'eeg_data_{participant_id}.csv'
    
    try:
        with open(input_filename, 'r') as infile, open(output_filename, 'a', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            for i, row in enumerate(reader):
                if i == 0 and row[0] != 'Participant ID':
                    writer.writerow(['Participant ID', 'Timestamp', 'TP9', 'AF7', 'AF8', 
                                   'TP10', 'Right AUX', 'Marker'])
                
                if row and row[0] != 'Participant ID':
                    writer.writerow(row + [marker])
    except FileNotFoundError:
        print(f"Warning: {input_filename} not found")

def update_eye_data_with_marker(marker):
    """Update eye tracking data file with marker"""
    participant_id = session.get('participant_id', 'unknown')
    input_filename = GameConfig.EYE_DATA_CSV
    output_filename = f'eye_data_with_marker/eye_data_{participant_id}.csv'
    
    try:
        with open(input_filename, 'r') as infile, open(output_filename, 'a', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            for i, row in enumerate(reader):
                if i == 0 and row[0] != 'Timestamp':
                    writer.writerow(['Timestamp', 'Timestamp (ISO 8601)', 'X', 'Y', 'Marker'])
                
                if row and row[0] != 'Timestamp':
                    writer.writerow(row + [marker])
    except FileNotFoundError:
        print(f"Warning: {input_filename} not found")

def log_user_interaction(event, trial, **kwargs):
    """Log user interaction to CSV file"""
    timestamp = datetime.now()
    participant_id = session.get('participant_id', 'unknown')
    
    with open(GameConfig.USER_INTERACTIONS_CSV, 'a', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        row = [participant_id, timestamp, event, trial]
        
        # Add all expected columns in order
        expected_columns = [
            'station_id', 'action', 'image_shown', 'attack', 'mobile_upi', 'charge_amount',
            'points_change', 'charge_cost', 'cyber_attack_loss', 'starting_charge', 'ending_charge',
            'station_1_status', 'station_1_image', 'station_1_attack', 'station_1_probabilities',
            'station_1_random_numbers', 'station_2_status', 'station_2_image', 'station_2_attack',
            'station_2_probabilities', 'station_2_random_numbers', 'station_3_status', 'station_3_image',
            'station_3_attack', 'station_3_probabilities', 'station_3_random_numbers', 'game_start_time',
            'game_end_time', 'charge_time', 'drive_time', 'decision_time', 'random_numbers',
            'chatbot_prompt',            
            'chatbot_suggested_kiosk'  
        ]
        
        for column in expected_columns:
            row.append(kwargs.get(column, ''))
        
        writer.writerow(row)


def append_interaction_data(station_id, kiosks_data):
    """
    Append a row to interaction_data.csv for the current station.
    kiosks_data: list of 3 dicts, each dict contains keys:
      SelectedImage, SelectedAttack, MobileUPI, ChargeAmount, PointsChange, ChargeCost, CyberAttackLoss, StartingCharge, EndingCharge
    """
    row = [station_id]
    for i in range(3):
        kiosk = kiosks_data[i]
        row.extend([
            kiosk.get('SelectedImage', ''),
            kiosk.get('SelectedAttack', ''),
            kiosk.get('MobileUPI', ''),
            kiosk.get('ChargeAmount', ''),
            kiosk.get('PointsChange', ''),
            kiosk.get('ChargeCost', ''),
            kiosk.get('CyberAttackLoss', ''),
            kiosk.get('StartingCharge', ''),
            kiosk.get('EndingCharge', ''),
        ])
    
    chatbot_response = kiosks_data[3].get('ChatbotResponse', '')
    chatbot_response = chatbot_response.replace('\n', ' ').replace('\r', ' ').replace(',', ' ')
    row.append(chatbot_response)  

    with open(GameConfig.INTERACTION_DATA_CSV, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)



def get_charging_page_sections():
    # Define rectangles as (x1_pct, y1_pct, x2_pct, y2_pct)
    return {        
        "kiosk_1": (0, 0, 0.23, 1.0),
        "kiosk_2": (0.23, 0.0, 0.46, 1.0),
        "kiosk_3": (0.46, 0.0, 0.69, 1.0),
        "battery_container": (0.69, 0.0, 1.0, 1.0),
    }

def point_in_rect(x, y, rect, width=1920, height=1080):
    x1_pct, y1_pct, x2_pct, y2_pct = rect
    x1, y1, x2, y2 = x1_pct * width, y1_pct * height, x2_pct * width, y2_pct * height
    return x >= x1 and x <= x2 and y >= y1 and y <= y2

def map_gaze_to_section(x, y, screen_width=1920, screen_height=1080):
    sections = get_charging_page_sections()
    for name, rect in sections.items():
        if name == "unknown":
            continue
        if point_in_rect(x, y, rect, screen_width, screen_height):
            return name
    return "unknown"

def map_numpy_gaze_array(coords, screen_w=1920, screen_h=1080, detect_normalized=True):
    """
    Map an array-like of gaze points to section names.
    coords: array-like shape (N,2) with columns [x, y].
      - If values look normalized (max <= 1.0) and detect_normalized is True,
        they will be scaled to pixels using screen_w/screen_h.
      - Otherwise treated as pixel coordinates.
    Returns: list of section names (length N).
    """
    
    a = np.asarray(coords)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("coords must be shape (N,2)")
    xs = a[:, 0].astype(float)
    ys = a[:, 1].astype(float)
    if detect_normalized:
        try:
            if np.nanmax(xs) <= 1.0 and np.nanmax(ys) <= 1.0:
                xs = xs * screen_w
                ys = ys * screen_h
        except Exception:
            pass
    sections = [map_gaze_to_section(float(x), float(y), screen_w, screen_h) for x, y in zip(xs, ys)]
    return sections

def gaze_data_to_section_percentage(gaze_data, screen_w=1920, screen_h=1080):
    section_counts = {
        "kiosk_1": 0,
        "kiosk_2": 0,
        "kiosk_3": 0,
        "battery_container": 0,
        "unknown": 0
    }
    total_count = 0
    
    sections = map_numpy_gaze_array(gaze_data, screen_w, screen_h)
    
    for section in sections:
        if section in section_counts:
            section_counts[section] += 1
        else:
            section_counts["unknown"] += 1
        total_count += 1
    
    if total_count == 0:
        return {k: 0.0 for k in section_counts.keys()}
    
    section_percentages = {k: v / total_count for k, v in section_counts.items()}
    return section_percentages
    

# ==================== INITIALIZATION ====================

# Initialize directories and files
ensure_directories_exist()
initialize_csv_files()
# Clear eye CSV every run (user requested)
clear_eye_data_file()
start_gaze_recorder_once()

# Initialize RAG store (persistent)
try:
    rag_store = RAGStore()
except Exception as e:
    print(f"RAG store initialization failed: {e}")
    rag_store = None

# Load game configuration
setup_variables = load_setup_variables()
deepfake_images, real_images = load_images()

# ==================== FLASK ROUTES ====================

@app.before_request
def before_request():
    """Initialize session variables before each request"""
    if 'help_click_count' not in session:
        session['help_click_count'] = 0

@app.route('/')
def consent():
    """Display consent form"""
    if 'participant_id' not in session:
        session['participant_id'] = str(datetime.now().timestamp()).replace('.', '')
    
    send_eeg_marker('consent_page_0')
    send_eye_marker('consent_page_0')
    return render_template('consent.html')

@app.route('/consent', methods=['POST'])
def handle_consent():
    """Handle consent form submission"""
    consent = request.form['consent']
    send_eeg_marker('consent_decision_0')
    send_eye_marker('consent_decision_0')
    log_user_interaction('consent_decision', 0, action=consent)
    
    if consent == 'agree':
        return redirect(url_for('demographics'))
    else:
        return redirect(url_for('consent_declined'))

@app.route('/demographics')
def demographics():
    """Display demographics form"""
    send_eeg_marker('demographics_page_0')
    send_eye_marker('demographics_page_0')
    return render_template('demographics.html')

@app.route('/demographics', methods=['POST'])
def handle_demographics():
    """Handle demographics form submission and initialize game"""
    # Store demographics
    session['demographics'] = {
        'age': request.form['age'],
        'gender': request.form['gender'],
        'education': request.form['education'],
        'specialization': request.form['specialization'],
        'ev_experience': request.form['ev_experience']
    }
    
    send_eeg_marker('demographics_submitted_0')
    send_eye_marker('demographics_submitted_0')
    log_user_interaction('demographics_submitted', 0, action=session['demographics'])
    
    # Save demographics to CSV
    with open(GameConfig.DEMOGRAPHICS_CSV, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([session['participant_id'], datetime.now()] + 
                       list(session['demographics'].values()))
    
    # Initialize game state
    session['trial'] = 1
    session['points'] = setup_variables['starting_points']
    session['charge'] = random.randint(int(setup_variables['initial_charge_min']), 
                                     int(setup_variables['initial_charge_max']))
    session['game_start_time'] = datetime.now().replace(tzinfo=None)
    
    return redirect(url_for('instructions'))

@app.route('/instructions')
def instructions():
    """Display game instructions"""
    send_eeg_marker('instructions_page_0')
    send_eye_marker('instructions_page_0')
    log_user_interaction('instructions_page', 0)
    return render_template('instructions.html')

@app.route('/index')
def index():
    """Main game interface - display charging stations"""
    trial = session.get('trial', 0)
    send_eeg_marker(f'index_page_{trial}')
    send_eye_marker(f'index_page_{trial}')
    start_time = datetime.now()

    session['chatbot_response'] = ""
    
    # Check if game should end
    if is_game_over():
        return redirect(url_for('end_game'))
    
    # Generate charging stations for this trial
    charging_stations = generate_charging_stations()
    session['stations'] = charging_stations
    
    # Calculate decision time
    decision_time = (datetime.now() - start_time).total_seconds()
    
    # Get query parameters
    show_skip_button = request.args.get('show_skip_button', 'false') == 'true'
    skip_charging = request.args.get('skip_charging') == 'true'
    
    # Handle skip charging logic
    if skip_charging:
        session['trial'] += 1
    
    # Calculate driving decrement for next phase
    session['driving_decrement'] = random.randint(
        int(setup_variables['driving_decrement_min']), 
        int(setup_variables['driving_decrement_max'])
    )
    
    # Get feedback from previous action
    feedback = session.pop('feedback', '')
    
    # Determine if images should be shown (based on coupon probability)
    image_prob = setup_variables['coupon_code_image_prob'] > random.random()
    
    log_user_interaction('index_page', trial, decision_time=decision_time)
    
    # After rendering the station interface, capture screenshot and caption
    # screenshot_path = capture_station_interface(trial)
    # station_caption = caption_image(screenshot_path)
    # print(f"Station {trial} interface caption: {station_caption}")

    # # Optionally, log or save the caption to a file or CSV
    # with open('station_captions.csv', 'a', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([trial, screenshot_path, station_caption])

    return render_template('index.html', 
                         stations=charging_stations,
                         show_skip_button=show_skip_button,
                         trial=session['trial'],
                         image_prob=image_prob,
                         points=session['points'],
                         feedback=feedback,
                         current_charge=int(session.get('charge', 0)))

@app.route('/start_charging', methods=['POST'])
def start_charging():
    """Handle charging station selection and process charging"""
    trial = session.get('trial', 0)
    station_id = int(request.form['station_id'])
    image_shown = request.form['image_shown']
    attack = request.form['attack']
    mobile_upi = request.form['mobile_upi']
    charge_amount = int(request.form['charge_amount'])
    discount_applied = request.form.get('discount_coupon') == 'true'
    
    send_eeg_marker(f'start_charging_{trial}')
    send_eye_marker(f'start_charging_{trial}')
    
    # Initialize variables
    feedback = ''
    points_change = 0
    charge_cost = charge_amount * setup_variables['cost_per_minute']
    starting_charge = session['charge']
    cyber_attack_loss = 0
    
    # Check if player can afford charging
    if not can_afford_charging(charge_amount):
        feedback = f'Insufficient points to charge for {charge_amount} minutes at station {station_id}'
        session['feedback'] = feedback
        return redirect(url_for('index'))
    
    # Process charging based on attack status
    if attack == 'Safe':
        feedback = f'Successfully charged at kiosk {station_id}'
        if discount_applied:
            charge_cost *= (1 - setup_variables['discount_coupon_percentage'])
            feedback += f'. Discount of {setup_variables["discount_coupon_percentage"] * 100}% applied.'
    else:
        feedback = f'Cyber attack encountered at kiosk {station_id}'
        # Reduce charge amount due to cyber attack
        charge_amount = int(charge_amount * setup_variables['cyber_attack_charge_reduction'])
        cyber_attack_loss = setup_variables['cyber_attack_loss_points']
        # feedback += f'. Lost {cyber_attack_loss} additional points due to cyber attack.'
    
    # Update points and charge
    session['points'] -= (charge_cost + cyber_attack_loss)
    session['points'] = max(session['points'], GameConfig.MIN_POINTS)
    
    # Calculate ending charge
    ending_charge = min(starting_charge + charge_amount * setup_variables['charge_rate_per_minute'], 
                       GameConfig.MAX_CHARGE)
    session['charge'] = ending_charge
    
    # Create detailed feedback
    detailed_feedback = (
        f"Starting charge: {int(starting_charge)}%. "
        f"Ending charge: {int(ending_charge)}%. "
        f"Charge cost: {int(charge_cost)} points. "
        f"Cyber attack loss: {int(cyber_attack_loss)} points."
    )
    
    feedback = f"{feedback} {detailed_feedback}"
    session['feedback'] = feedback
    # session['charging_time'] = charge_amount
    session['charging_time'] = 1

    session["starting_charge"] = starting_charge
    session["ending_charge"] = ending_charge
    

    # Log the interaction
    log_user_interaction('start_charging', trial, 
                        station_id=station_id, 
                        action='start_charging',
                        image_shown=image_shown, 
                        attack=attack, 
                        charge_amount=charge_amount,
                        points_change=points_change, 
                        charge_cost=charge_cost, 
                        cyber_attack_loss=cyber_attack_loss,
                        starting_charge=starting_charge, 
                        ending_charge=ending_charge,
                        chatbot_response = "",
                        random_numbers=session['stations'][station_id-1]['random_numbers'])
    print("DEBUG:", starting_charge, ending_charge, charge_amount)


    # Load screenshot_history from file
    screenshot_history_path = get_screenshot_history_path()
    screenshot_history = load_json_file(screenshot_history_path, {})
    desc_json = screenshot_history.get(str(trial))
    kiosk_visual = {}
    if desc_json:
        try:
            desc_json_obj = json.loads(str(desc_json))
            kiosk_key = f"Kiosk {station_id}"
            kiosk_visual = desc_json_obj.get(kiosk_key, {})
        except Exception:
            kiosk_visual = {}

    charging_event = {
        "trial": trial,
        "kiosk": station_id,
        "image": image_shown,
        "attack": attack,
        "cyber_attacked": attack == "Cyber Attack",
        "charge_amount": charge_amount,
        "points_after": session['points'],
        "ending_charge": session['charge'],
        "visual_content": kiosk_visual.get("Visual Content", ""),
        "anomalies": kiosk_visual.get("Anomalies", "")
    }

    # Store charging_context in file
    charging_context_path = get_charging_context_path()
    charging_context = load_json_file(charging_context_path, [])
    charging_context.append(charging_event)
    save_json_file(charging_context_path, charging_context)

    # Persist charging event into RAG store for later retrieval
    try:
        if rag_store is not None:
            doc_text = (
                f"Station {trial} Kiosk {station_id} - Image: {image_shown}. "
                f"Attack: {attack}. CyberAttacked: {attack == 'Cyber Attack'}. "
                f"Visual: {charging_event.get('visual_content','')}. Anomalies: {charging_event.get('anomalies','')}"
            )
            rag_doc = {
                'id': f"c_{trial}_{station_id}",
                'kiosk': f"Kiosk {station_id}",
                'text': doc_text,
                'meta': {
                    'trial': trial,
                    'source': 'start_charging',
                    'cyber_attacked': bool(charging_event.get('cyber_attacked', False))
                }
            }
            rag_store.add_documents([rag_doc])
    except Exception as e:
        print(f"RAG indexing failed during start_charging: {e}")

    # After processing charging, build kiosks_data for all kiosks in this station
    kiosks_data = []
    for i in range(1, 4):
        kiosks_data.append({
            'SelectedImage': request.form.get(f'kiosk{i}_image', ''),
            'SelectedAttack': request.form.get(f'kiosk{i}_attack', ''),
            'MobileUPI': mobile_upi if station_id == i else '',
            'ChargeAmount': charge_amount if station_id == i else '',
            'PointsChange': 0,
            'ChargeCost': charge_cost if station_id == i else '',
            'CyberAttackLoss': cyber_attack_loss if station_id == i else '',
            'StartingCharge': session.get('starting_charge', '') if station_id == i else '',
            'EndingCharge': session.get('ending_charge', '') if station_id == i else '',
        })

    kiosks_data.append({'ChatbotResponse': session.get('chatbot_response', '')})
    # Immediately append this station's data to interaction_data.csv
    append_interaction_data(trial, kiosks_data)

    # Check if game should continue
    if is_game_over():
        return redirect(url_for('end_game'))
    
    # Handle keep screen option
    keep_screen = request.form.get('keep_screen') == 'true'
    if keep_screen:
        return redirect(url_for('charging', keep_screen='true'))
    
    # return redirect(url_for('charging'))
    return redirect(url_for('charging'))

@app.route('/charging')
def charging():
    """Display charging animation/progress"""
    trial = session.get('trial', 0)
    send_eeg_marker(f'charging_page_{trial}')
    send_eye_marker(f'charging_page_{trial}')
    
    charge_time = session.get('charging_time', 0)
    starting_charge = session.get('starting_charge',0)
    ending_charge = session.get('ending_charge', 0)
    charge_amount = ending_charge - starting_charge
    log_user_interaction('charging_page', trial, charge_time=charge_time)
    print("DEBUG:", starting_charge, ending_charge, charge_time, charge_amount)

    return render_template('charging.html', charging_time=charge_time,
                           current_charge=starting_charge,charge_amount=charge_amount)

@app.route('/post_charging', methods=['POST'])
def post_charging():
    """Handle post-charging help requests"""
    help_text = request.json.get('help_text')
    trial_number = request.json.get('trial')
    participant_id = session.get('participant_id', 'unknown')
    
    send_eeg_marker(f'after_charging_page_{trial_number}')
    send_eye_marker(f'after_charging_page_{trial_number}')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(GameConfig.HELP_REQUESTS_CSV, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([participant_id, timestamp, help_text, trial_number])
    
    return jsonify({'status': 'success'})

@app.route('/disabled_charging')
def disabled_charging():
    """Display charging interface when charging is disabled"""
    trial = session.get('trial', 0)
    send_eeg_marker(f'disabled_charging_{trial}')
    send_eye_marker(f'disabled_charging_{trial}')
    start_time = datetime.now()
    
    # Generate charging stations (same logic as index)
    charging_stations = generate_charging_stations()
    session['stations'] = charging_stations
    
    feedback = session.pop('feedback', '')
    decision_time = (datetime.now() - start_time).total_seconds()
    
    # Get parameters
    show_skip_button = request.args.get('show_skip_button', 'false') == 'true'
    # image_prob = setup_variables['coupon_code_image_prob'] > random.random()
    
    log_user_interaction('disabled_charging_page', trial, decision_time=decision_time)
    
    return render_template('disabled_charging.html',
                         stations=charging_stations,
                         show_skip_button=show_skip_button,
                         trial=session['trial'],
                        #  image_prob=image_prob,
                         points=session['points'],
                         feedback=feedback,
                         current_charge=session['charge'])

@app.route('/driving')
def driving():
    """Handle driving phase and charge depletion"""
    trial = session.get('trial', 0)
    send_eeg_marker(f'driving_page_{trial}')
    send_eye_marker(f'driving_page_{trial}')
    
    # Calculate driving charge loss
    driving_decrement = random.randint(
        int(setup_variables['driving_decrement_min']), 
        int(setup_variables['driving_decrement_max'])
    )
    session['driving_decrement'] = driving_decrement
    
    # Update charge after driving
    ending_charge = session['charge']
    session['charge'] = max(ending_charge - driving_decrement, GameConfig.MIN_CHARGE)
    
    log_user_interaction('driving_page', trial, drive_time=driving_decrement)
    
    print(f"Driving - Decrement: {driving_decrement}, New charge: {session['charge']}")
    
    # Check if charge is depleted
    if session['charge'] == 0:
        send_eeg_marker(f'low_charge_{trial}')
        send_eye_marker(f'low_charge_{trial}')
        return render_template('low_charge.html')
    
    # Get skip charging parameter
    skip_charging = request.args.get('skip_charging') == 'true'
    
    return render_template('driving.html', 
                         driving_decrement=driving_decrement, 
                         skip_charging=skip_charging, current_charge=ending_charge, )

@app.route('/end_game')
def end_game():
    """Display game end screen with results"""
    trial = session.get('trial', 0)
    send_eeg_marker(f'end_game_{trial}')
    send_eye_marker(f'end_game_{trial}')
    
    points = session.get('points', 0)
    end_time = datetime.now().replace(tzinfo=None)
    game_start_time = session.get('game_start_time', end_time)
    
    # Handle timezone issues
    if game_start_time.tzinfo is None:
        game_start_time = game_start_time.replace(tzinfo=timezone.utc)
    
    game_duration = (end_time.replace(tzinfo=timezone.utc) - game_start_time).total_seconds()
    
    log_user_interaction('end_game', trial, points=points, game_duration=game_duration)
    
    return render_template('end_game.html', points=points)

@app.route('/restart')
def restart():
    """Restart the game"""
    send_eeg_marker('restart_game_0')
    send_eye_marker('restart_game_0')
    session.clear()  # Clear all session data
    return redirect(url_for('consent'))

@app.route('/consent_declined')
def consent_declined():
    """Display consent declined message"""
    send_eeg_marker('consent_declined_0')
    send_eye_marker('consent_declined_0')
    return render_template('consent_declined.html')

# ==================== API ROUTES ====================

@app.route('/get_image/<filename>')
def get_image(filename):
    """Serve images from appropriate folder"""
    if 'deepfake' in filename.lower():
        folder = GameConfig.DEEPFAKE_FOLDER
    else:
        folder = GameConfig.REAL_FOLDER
    return send_from_directory(folder, filename)

@app.route('/eeg_marker')
def eeg_marker():
    """Get EEG marker from session"""
    marker = session.pop('eeg_marker', None)
    return jsonify({'marker': marker})

@app.route('/eye_marker')
def eye_marker():
    """Get eye tracking marker from session"""
    marker = session.pop('eye_marker', None)
    return jsonify({'marker': marker})

@app.route('/kiosk_status', methods=['GET'])
def kiosk_status():
    """
    For each kiosk, calculate weighted cyber attack probability:
    - 60% from image attack probability (across all stations/kiosks)
    - 40% from kiosk attack probability at this station (same kiosk number and station id, regardless of image)
    Returns: {kiosk_id: {'image': ..., 'station_id': ..., 'weighted_attack_probability': ...}}
    Always returns all 3 kiosks, even if occupied.
    """
    stations = session.get('stations', [])
    station_id = session.get('trial', '')  # Use the current trial as the station id for all kiosks
    result = {}
    try:
        with open(GameConfig.INTERACTION_DATA_CSV, 'r') as file:
            reader = list(csv.DictReader(file))
        for k in range(1, 4):
            # Check if kiosk is occupied
            status = stations[k-1]['status'] if stations and len(stations) >= k else ''
            if status == 'Occupied':
                result[k] = {
                    'station_id': station_id,
                    'weighted_attack_probability': 0.0,
                    'image_attack_probability': 0.0,
                    'kiosk_station_attack_probability': 0.0,
                    'Status': status
                }
                continue
            image = stations[k-1]['image'] if stations and len(stations) >= k else ''
            image = image.lower().strip()
            # Image-based attack stats
            image_total = 0
            image_attacks = 0
            # Kiosk-at-this-station attack stats (regardless of image)
            kiosk_total = 0
            kiosk_attacks = 0
            for row in reader:
                # Image-based stats
                for ki in range(1, 4):
                    img_val = row.get(f'Kiosk{ki}_SelectedImage', '').lower().strip()
                    attack_val = row.get(f'Kiosk{ki}_SelectedAttack', '').lower().strip()
                    if img_val and img_val == image:
                        image_total += 1
                        if attack_val == 'cyber attack':
                            image_attacks += 1
                # Kiosk-at-this-station stats (regardless of image)
                station_id_val = row.get('StationID', '').strip()
                kiosk_attack_val = row.get(f'Kiosk{k}_SelectedAttack', '').lower().strip()
                if str(station_id_val) == str(station_id):
                    kiosk_total += 1
                    if kiosk_attack_val == 'cyber attack':
                        kiosk_attacks += 1
            image_prob = (image_attacks / image_total) if image_total > 0 else 0.0
            kiosk_prob = (kiosk_attacks / kiosk_total) if kiosk_total > 0 else 0.0
            weighted_prob = round(0.6 * image_prob + 0.4 * kiosk_prob, 2)
            # Debug output
            #print(f"Kiosk {k} | Station {station_id} | Image '{image}' | image_total={image_total} image_attacks={image_attacks} kiosk_total={kiosk_total} kiosk_attacks={kiosk_attacks}")
            result[k] = {
                'station_id': station_id,
                'weighted_attack_probability': weighted_prob,
                'image_attack_probability': round(image_prob,2),
                'kiosk_station_attack_probability': round(kiosk_prob,2),
                'Status': status
            }
            #print(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chatbot')
def chatbot():
    """Display chatbot interface"""
    # Gemini is default, so put it first in the list
    bots = list(chatbots.keys())
    if 'gemini' in bots:
        bots.remove('gemini')
        bots = ['gemini'] + bots
    return render_template('chatbot.html', bots=bots, default_bot='gemini')


@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle sending message to the selected chatbot and return station colors."""
    bot_name = request.json.get("bot") or "gemini"
    message = request.json.get("message")

    # Deduct 5 points for LLM usage
    points = session.get('points', 0)
    points = max(points - 5, GameConfig.MIN_POINTS)
    session['points'] = points

    if bot_name not in chatbots.keys():
        return jsonify({"error": "Bot not configured.", "points": points}), 400


    # Get kiosk status details as a dict
    resp = kiosk_status()
    # if hasattr(resp, 'json'):
    #     kiosks_data = resp.json
    # else:
    #     kiosks_data = resp.get_json()
    kiosks_data = {}

    # Always return three yellows for colors (Kiosk 1, 2, 3)
    colors = {"1": "yellow", "2": "yellow", "3": "yellow"}

    # If user message is 'checknow', return all kiosk details
    if isinstance(message, str) and message.strip().lower() == 'checknow':
        return jsonify({"reply": f"Kiosk details: {kiosks_data}", "colors": colors, "points": points})




    # Compose a system prompt for the LLM to guide its behavior
    green = KioskThresholds.GREEN
    yellow = KioskThresholds.YELLOW
    help_agent_prompt = (
        f"You are a Help Agent for EV charging kiosks. Based on cyber attack risk, recommend the safest available kiosk for the user to use. Always address user directly. "
        f"In this scenario, user has to cross 20 stations and is at {session.get('trial',0)} station, and around 15% charge is reduced per station due to driving (Do not mention these numbers directly). The User can charge at kiosks using points (1 point = 1 charge) ."
        "Never recommend a kiosk that is occupied or unavailable. If all kiosks are risky, suggest the least risky available one or you may suggest to skip (please suggest to skip rarely, only when user has sufficient charge). "
        "Be brief (50 - 100 words), clear, Only mention the kiosk(s) and reason why they should or should not be chosen. "
        "Also give a small explanation of your recommendation based on the descriptions below. "
        "Following this you will be given previous kiosk description which the user picked and whether user had faced cyberattack. "
        "This will be followed by the UI description of the current 3 kiosk. Similar kiosk will have similar cyberattack chances. \n"
    )

    while(SimulationConfig.ScreenshotProcessing):
        pass


    # Load charging_context from file
    charging_context_path = get_charging_context_path()
    charging_context = load_json_file(charging_context_path, [])
    context_lines = []
    for event in charging_context:
        context_lines.append(
            f"Station {event['trial']} (Kiosk {event['kiosk']}): "
            f"Attack: {event['attack']}, "
            f"Cyberattacked: {event['cyber_attacked']}, "
            f"Visual: {event.get('visual_content', '')}, "
            f"Anomalies: {event.get('anomalies', '')}, "
            f"Charge: {event['charge_amount']}, "
            f"Points after: {event['points_after']}, "
            f"Ending charge: {event['ending_charge']}"
        )
    context_str = "\n".join(context_lines)

    screenshot_context = get_screenshot_context()
    full_prompt = help_agent_prompt
    if context_str:
        full_prompt += f"Previous station context:\n{context_str}\n\n"
    if screenshot_context:
        full_prompt += f"Screenshot Analysis: {screenshot_context}\n\n"

    # RAG retrieval: for each kiosk, retrieve top-3 similar stored docs above threshold
    if(SimulationConfig.RAG):
        try:
            rag_retrievals = {}
            if rag_store is not None:
                threshold = 0.35  # baseline similarity
                for kiosk_id in ['Kiosk 1', 'Kiosk 2', 'Kiosk 3']:
                    # build query from screenshot_context if available for the kiosk, else use kiosk label
                    if screenshot_context:
                        try:
                            sc_obj = json.loads(screenshot_context)
                            info = sc_obj.get(kiosk_id, {})
                            query_text = f"Visual Content: {info.get('Visual Content','')} Anomalies: {info.get('Anomalies','')}"
                        except Exception:
                            query_text = kiosk_id
                    else:
                        query_text = kiosk_id

                    # retrieve candidates from RAG
                    candidates = rag_store.search(query_text, top_k=10, score_threshold=threshold)
                    # filter to only charging events from same kiosk
                    filtered = []
                    for meta, score in candidates:
                        try:
                            m = meta.get('meta', {})
                            source = m.get('source')
                            kiosk_label = meta.get('kiosk')
                            if source == 'start_charging' and kiosk_label == kiosk_id:
                                filtered.append((meta, score))
                        except Exception:
                            continue
                    # keep top 3
                    rag_retrievals[kiosk_id] = filtered[:3]

                # Append retrieved charging-event contexts only
                full_prompt += "Retrieved past charging events (same kiosk) from RAG (most relevant first):\n"
                for k, items in rag_retrievals.items():
                    if not items:
                        continue
                    full_prompt += f"{k}:\n"
                    for meta, score in items:
                        trial_meta = meta.get('meta', {}).get('trial', '?')
                        cyber_flag = meta.get('meta', {}).get('cyber_attacked', False)
                        full_prompt += f"- Score: {score:.2f} | Trial: {trial_meta} | CyberAttacked: {cyber_flag} | Text: {meta.get('text','')}\n"
                full_prompt += "\n"
        except Exception as e:
            print(f"RAG retrieval failed in send_message: {e}")


    if (SimulationConfig.GazeTracking):
        gaze_data = []

        with gaze_context_lock:
            gaze_data = gaze_context['context'].copy()
        
        if not gaze_data:
            print("error: no gaze data available for chatbot")
        else:
            gaze_percentage = gaze_data_to_section_percentage(np.array(gaze_data))
            print("Gaze section percentages:", gaze_percentage)
            full_prompt += "\n Below given will be the distribution of user gaze across different sections of the screen while they were viewing the kiosks. More a user viewed a section, more likely is it that user wants charge at that station or user has found an anomaly in that section.\n"
            full_prompt += "Use this information to help guide your recommendation of the safest kiosk to use. Do mention something regarding it so that user that you have a better understanding\n"
            full_prompt += "\nUser gaze distribution across screen sections:\n"
            full_prompt += f"Kiosk 1: {gaze_percentage.get('kiosk_1',0)*100:.2f}%\n"
            full_prompt += f"Kiosk 2: {gaze_percentage.get('kiosk_2',0)*100:.2f}%\n"
            full_prompt += f"Kiosk 3: {gaze_percentage.get('kiosk_3',0)*100:.2f}%\n"
            full_prompt += f"Battery Container & Points: {gaze_percentage.get('battery_container',0)*100:.2f}%\n"



        

    full_prompt += f"\nCurrent Charge = {int(session.get('charge',0))}%\n"
    full_prompt += f"\nCurrent Points = {points}\n"
    full_prompt += f"\nCurrent Station Number = {session.get('trial',0)}\n"

    full_prompt += f"User question: {message}"
    print("----------------------------------------------------")
    for i in range(0, len(full_prompt), 500):
        print(full_prompt[i:i+500])
    print("----------------------------------------------------")


    #TODO: set suggested_kiosk based on LLM response
    suggested_kiosk = "None"

    #print("Full prompt to LLM:", full_prompt)

    # Use LLM to answer
    if bot_name == "gemini":
        llm = ChatGoogleGenerativeAI(
            google_api_key=chatbots["gemini"]["api_key"],
            model=chatbots["gemini"]["model"]
        )
        try:
            answer = llm.invoke(full_prompt)
            reply_text = getattr(answer, 'content', str(answer))
            trial = session.get('trial', 0)
            log_user_interaction(
                event='chatbot_interaction',
                trial=trial,
                chatbot_prompt=message,
                chatbot_suggested_kiosk=suggested_kiosk
            )
            session['chatbot_response'] = reply_text
            return jsonify({"reply": reply_text, "colors": colors, "points": points})
        except Exception as e:
            return jsonify({"error": str(e), "points": points}), 500
    elif bot_name == "openai":
        llm = ChatOpenAI(
            openai_api_key=chatbots["openai"]["api_key"],
            model=chatbots["openai"]["model"]
        )
        try:
            answer = llm.invoke(full_prompt)
            reply_text = getattr(answer, 'content', str(answer))
            trial = session.get('trial', 0)
            log_user_interaction(
                event='chatbot_interaction',
                trial=trial,
                chatbot_prompt=message,
                chatbot_suggested_kiosk=suggested_kiosk
            )
            return jsonify({"reply": reply_text, "colors": colors, "points": points})
        except Exception as e:
            return jsonify({"error": str(e), "points": points}), 500
    else:
        return jsonify({"error": "Bot not configured.", "points": points}), 400


@app.route('/help_click', methods=['POST'])
def help_click():
    """Handle help button clicks"""
    session['help_click_count'] = session.get('help_click_count', 0) + 1
    return jsonify(click_count=session['help_click_count'])

@app.route('/help_submit', methods=['POST'])
def help_submit():
    """Handle help text submission"""
    data = request.json
    help_text = data.get('help_text')
    trial = data.get('trial')
    participant_id = session.get('participant_id', 'unknown')
    
    # Save help request to CSV
    with open(GameConfig.HELP_REQUESTS_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([participant_id, timestamp, help_text, trial])
    
    return jsonify(success=True)

# Install: pip install selenium pillow
from selenium import webdriver
from PIL import Image

def capture_station_screenshot(url, save_path):
    driver = webdriver.Chrome()  # Or use your preferred browser
    driver.get(url)
    driver.save_screenshot(save_path)
    driver.quit()

def capture_station_interface(trial):
    """Capture screenshot of the station interface and return the image path."""
    url = f"http://localhost:5000/index?trial={trial}"
    save_path = f"station_interface_{trial}.png"
    driver = webdriver.Chrome()
    driver.get(url)
    driver.save_screenshot(save_path)
    driver.quit()
    return save_path

@app.route('/upload_screenshot', methods=['POST'])

def upload_screenshot():
    SimulationConfig.ScreenshotProcessing = True
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    trial = data.get('trial', 0)
    image_bytes = base64.b64decode(image_data)
    os.makedirs('station_screenshots', exist_ok=True)
    image_path = f'station_screenshots/station_interface_{trial}.png'
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    # Generate description using Gemini Vision
    prompt = """
    You are an expert in analyzing EV charging network user interfaces for signs of potential cyberattacks.
    Analyze the given interface carefully.
    For each Kiosk section (Kiosk 1, Kiosk 2, Kiosk 3):
    Describe the visual content shown in its promotion or coupon area (images, graphics, text, faces, QR codes, etc.).
    Identify any anomalies or signs of cyberattack, such as deepfake or AI-generated images, unrealistic discounts, phishing prompts, or suspicious branding.

    If a kiosk's status is Occupied, simply output "Visual Content": "Occupied" and "Anomalies": "None".
    If no images or graphics appear, write "Visual Content": "No visual content available".

    Return the result strictly in JSON format as:
    {
        "Kiosk 1": {
            "Visual Content": "...",
            "Anomalies": "..."
        },
        "Kiosk 2": {
            "Visual Content": "...",
            "Anomalies": "..."
        },
        "Kiosk 3": {
            "Visual Content": "...",
            "Anomalies": "..."
        }
    }
    """
    description = generate_caption_gemini(image_path, prompt)
    # Store screenshot description per trial in session
    try:
        trial_int = int(trial)
    except Exception:
        trial_int = trial

    # Save screenshot_history to file
    screenshot_history_path = get_screenshot_history_path()
    screenshot_history = load_json_file(screenshot_history_path, {})
    description_cleaned = description.strip().strip('`')[4:]
    screenshot_history[str(trial_int)] = description_cleaned
    save_json_file(screenshot_history_path, screenshot_history)
    set_screenshot_context(description)

    # NOTE: we DO NOT index raw screenshot descriptions into the RAG store here.
    # The RAG database is intended to hold only charging events 
    SimulationConfig.ScreenshotProcessing = False
    return jsonify({'description': description})



# ==================== MAIN APPLICATION ====================

if __name__ == '__main__':
    print("="*60)
    print("EV CHARGING SIMULATION GAME")
    print("="*60)
    print(f"Starting points: {setup_variables['starting_points']}")
    print(f"Number of trials: {setup_variables['number_of_trials']}")
    print(f"Deepfake images loaded: {len(deepfake_images)}")
    print(f"Real images loaded: {len(real_images)}")
    print("="*60)
    app.run(debug=False, use_reloader=False)



def save_station_to_interaction_data(trial, stations):
    """
    Save the current station's kiosk data to interaction_data.csv.
    trial: current station number (1-based)
    stations: list of 3 dicts, each dict contains keys:
      SelectedImage, SelectedAttack, MobileUPI, ChargeAmount, PointsChange, ChargeCost, CyberAttackLoss, StartingCharge, EndingCharge
    """
    append_interaction_data(trial, stations)

