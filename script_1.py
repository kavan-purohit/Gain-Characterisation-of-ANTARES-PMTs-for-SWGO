"""
===============================================================

     This script acquire and record the PMT data
     and store in an HDF5 file

===============================================================
"""

import numpy as np
from modules import inst_controll as inst
import time
import sys
import matplotlib.pyplot as plt
import h5py
import datetime
import os

# Folder where all waveform files (HDF5) will be saved
folder = "DataNew/"

# ----------------------------------------------------------------------
# Parse command-line arguments: serial number (string) and high voltage (int)
# ----------------------------------------------------------------------
try:
    SN = sys.argv[1]                  # PMT serial number
    HV = int(sys.argv[2])            # High voltage applied to the PMT
except IndexError:
    # Triggered if user does not pass both arguments
    print('Usage: python {} <serial number> <high voltage>'.format(sys.argv[0]))
    sys.exit()
except TypeError:
    # Triggered if HV cannot be converted to an integer
    print('Usage: python {} <serial number> <high voltage>, high voltage must be an integer'.format(sys.argv[0]))
    sys.exit()

# Optional argument: number of pulses to record
try:
    number = int(sys.argv[3])
except (IndexError, ValueError):
    number = 10000                   # Default number of samples if user doesn't specify

# ----------------------------------------------------------------------
# Initialize and configure the function generator
# ----------------------------------------------------------------------
# The function generator produces a known pulse for testing PMT response.
# Parameters currently set: 1 kHz repetition rate, 1 V amplitude, ~500 µs pulse width.
func = inst.Funk_Gen()
func.pulse(1e3, 1, width=5.00195e-4, channel=1)
func.on()

# ----------------------------------------------------------------------
# Initialize the oscilloscope interface (Tektronix MSO)
# ----------------------------------------------------------------------
osci = inst.Osci()
time.sleep(2)     # Allow scope to finish remote initialization

# ----------------------------------------------------------------------
# Acquire waveforms from the oscilloscope
# ----------------------------------------------------------------------
# messung() returns:
#   y_values         -> raw ADC waveform values (CH1 only)
#   Measurement_time -> total time window per waveform
#   YOFF, YMU        -> vertical offset & vertical scaling from scope (convert ADC → volts)
#   samplerate       -> sampling rate in Hz
print("Starting measurement...")
y_values, Measurement_time, YOFF, YMU, samplerate = osci.messung(
    number, Measurement_time=200e-9, samplerate=6.25e9, chanels=['CH1']
)
print("Measurement completed.")

# ----------------------------------------------------------------------
# Plot all captured waveforms for a quick visual check (stacked waveforms)
# ----------------------------------------------------------------------
fig, ax = plt.subplots()
for waveform in y_values[0]:
    # Convert ADC values to volts: V = raw * gain + offset
    ax.plot(np.linspace(0, Measurement_time, len(waveform)), waveform * YMU + YOFF)

title = f'Captured Pulses SN={SN} HV={HV}V'
ax.set_title(title, fontsize=30)
ax.set_xlabel('Time (s)', fontsize=30)
ax.set_ylabel('Amplitude (V)', fontsize=30)
plt.show()

# ----------------------------------------------------------------------
# Ask user before saving: important to avoid saving bad measurements
# If selected 'n', the script will not save the data and exits.
# ----------------------------------------------------------------------
response = input("Are you satisfied with the earlier shown plot? Do you want to continue with saving the data? (y/n): ").strip().lower()
if response != 'y':
    print("Process canceled. Exiting without saving.")
    sys.exit()

# ----------------------------------------------------------------------
# Prepare filename for saving inside an HDF5 structure:
# file: <SN>.h5
# group names: HV(1), HV(2), HV(3), ...
# ----------------------------------------------------------------------
filename = f"{folder}{SN}.h5"

# Timestamp for tracking measurement sessions
timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ----------------------------------------------------------------------
# Save measurement results into hierarchical HDF5 structure
# ----------------------------------------------------------------------
with h5py.File(filename, 'a') as hf:
    # To ensure HV group does not overwrite older measurements,
    # Always add a suffix for the group name, starting from 1
    # Example: "1200(1)", "1200(2)", ...
    suffix = 1
    while f"{HV}({suffix})" in hf:
        suffix += 1
    group_name = f"{HV}({suffix})"

    print(f"Saving data under group: {group_name}")

    # Create storage groups inside this measurement entry
    waveforms_group = hf.create_group(f"{group_name}/waveforms")
    wf_info = hf.create_group(f"{group_name}/waveform_info")

    # Save CH1-only raw integer waveforms (2D dataset)
    separate_group = hf.create_group(f"{group_name}/separate_waveform")
    separate_group.create_dataset("data", data=y_values[0], dtype=np.int8)

    # Save raw waveform array including channel dimension
    waveforms_group.create_dataset("data", data=y_values, dtype=np.int8)

    # Save metadata necessary to reconstruct the physical waveform
    wf_info["samplerate"] = samplerate
    wf_info["v_gain"] = YMU
    wf_info["measurement_time"] = Measurement_time
    wf_info["y_off"] = YOFF
    wf_info["timestamp"] = timestamp
    wf_info["serial_number"] = SN

print(f"Data has been saved to '{filename}'")
