import numpy as np
import  pywt

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from collections import deque
from datetime import datetime
import streamlit as st
import websockets
import asyncio
import json
import threading

# ------------------------------
# Fault Severity Definitions
# ------------------------------
FAULT_THRESHOLDS = {
    "unbalance": 0.7,
    "misalignment": 1.2,
    "bearing_fault": 1.5,
    "rotor_fault": 2.0
}

FAULT_SEVERITY_LEVELS = {
    "Normal": (0, 0.3),
    "Mild": (0.3, 0.7),
    "Moderate": (0.7, 1.5),
    "Severe": (1.5, float("inf"))
}

# ------------------------------
# Historical Data Management
# ------------------------------
HISTORICAL_DATA = deque(maxlen=100)  # Store up to 100 data points

# ------------------------------
# Signal Analysis Functions
# ------------------------------

def calculate_rms(signal):
    """Calculate Root Mean Square (RMS) value of the signal."""
    return np.sqrt(np.mean(signal ** 2))


def perform_fft(signal, sampling_rate):
    """Perform FFT to extract frequency domain characteristics."""
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_values = np.abs(fft(signal))
    return freqs[:n // 2], fft_values[:n // 2]


def perform_dwt_analysis(signal, wavelet='db4', max_level=5):
    """Perform Discrete Wavelet Transform (DWT) analysis."""
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    energy_levels = [np.sum(np.square(c)) for c in coeffs]
    total_energy = np.sum(energy_levels)
    normalized_energy = [e / total_energy for e in energy_levels]

    features = {
        "Approximation Coefficients Energy": energy_levels[0],
        "Detail Coefficients Energy": energy_levels[1:],
        "Normalized Energy Distribution": normalized_energy
    }
    return coeffs, features


def classify_dwt_fault(features):
    """Classify fault based on DWT energy distribution."""
    detail_energy = np.sum(features["Detail Coefficients Energy"])
    if detail_energy < 0.3:
        return "Normal"
    elif detail_energy < 0.7:
        return "Unbalance"
    elif detail_energy < 1.2:
        return "Misalignment"
    elif detail_energy < 1.5:
        return "Bearing Fault"
    else:
        return "Rotor Fault"


def classify_faults(rms_value):
    """Classify fault based on RMS value."""
    for severity, (lower, upper) in FAULT_SEVERITY_LEVELS.items():
        if lower <= rms_value < upper:
            return severity
    return "Unknown"


def analyze_vibration_data(vibration_data, sampling_rate):
    """Analyze vibration data using RMS, FFT, and DWT."""
    rms_value = calculate_rms(vibration_data)
    fault_class_rms = classify_faults(rms_value)

    freqs, fft_values = perform_fft(vibration_data, sampling_rate)

    dwt_coeffs, dwt_features = perform_dwt_analysis(vibration_data)
    fault_class_dwt = classify_dwt_fault(dwt_features)

    dominant_frequency = freqs[np.argmax(fft_values)]

    severity = fault_class_dwt if fault_class_dwt != "Normal" else fault_class_rms

    return {
        "RMS Value": rms_value,
        "Fault Classification (RMS)": fault_class_rms,
        "Fault Classification (DWT)": fault_class_dwt,
        "Dominant Frequency": dominant_frequency,
        "Fault Severity": severity,
        "DWT Features": dwt_features
    }

# ------------------------------
# WebSocket Server
# ------------------------------

async def vibration_data_receiver(websocket, path):
    """Receive vibration data via WebSocket and process it."""
    try:
        async for message in websocket:
            data = json.loads(message)
            vibration_data = np.array(data['vibration_data'])
            sampling_rate = data['sampling_rate']

            # Analyze the received vibration data
            analysis_results = analyze_vibration_data(vibration_data, sampling_rate)

            # Update historical data
            update_historical_data(analysis_results)

            # Send analysis results back to the client (optional)
            await websocket.send(json.dumps(analysis_results))

    except Exception as e:
        print(f"Error processing WebSocket data: {e}")


def start_websocket_server():
    """Start the WebSocket server in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = websockets.serve(vibration_data_receiver, "192.168.137.124", 8765)  # Use your NodeMCU IP address here
    loop.run_until_complete(server)
    loop.run_forever()

# ------------------------------
# Streamlit Application
# ------------------------------

# Streamlit app setup
st.title("Induction Motor Vibration Analysis")
st.sidebar.header("Configuration")

# Sampling rate input
sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", min_value=1, max_value=10000, value=1600)

# Initialize WebSocket server in a background thread
if not hasattr(st, "websocket_thread"):
    st.websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    st.websocket_thread.start()

# Display instructions
st.info("Connect to the WebSocket server at ws://192.168.137.124:8765 to send vibration data for analysis.")  # Updated to use NodeMCU IP

# Display historical trends
if HISTORICAL_DATA:
    st.subheader("Historical Trends")
    timestamps = [entry["Timestamp"] for entry in HISTORICAL_DATA]
    rms_values = [entry["RMS Value"] for entry in HISTORICAL_DATA]
    severities = [entry["Fault Severity"] for entry in HISTORICAL_DATA]

    plt.figure(figsize=(12, 6))

    # Plot RMS trends
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, rms_values, marker='o', label="RMS Value")
    plt.title("Historical RMS Trend")
    plt.ylabel("RMS Value")
    plt.grid()
    plt.legend()

    # Plot fault severity trends
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, severities, marker='o', label="Fault Severity")
    plt.title("Historical Fault Severity Trend")
    plt.ylabel("Fault Severity")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)

# ------------------------------
# Historical Data Management
# ------------------------------

def update_historical_data(analysis_results):
    """Update historical data storage."""
    timestamp = datetime.now()
    historical_entry = {
        "Timestamp": timestamp,
        "RMS Value": analysis_results["RMS Value"],
        "Fault Severity": analysis_results["Fault Severity"],
        "Dominant Frequency": analysis_results["Dominant Frequency"]
    }
    HISTORICAL_DATA.append(historical_entry)
