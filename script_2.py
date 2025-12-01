"""
===============================================================

     This script loads previously recorded PMT waveforms stored
     in an HDF5 file created by script_1.py
     used only to view the metadata and waveforms

===============================================================
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import modules.data_analysis as data


# Specify the filename of the HDF5 file to load
# (Update this path depending on which PMT is being analyzed!)
filename = 'DataNew/TM0007.h5'  # Use the correct path

# ----------------------------------------------------------------------
# Opening the HDF5 file and inspect its structure
# ----------------------------------------------------------------------
with h5py.File(filename, 'r') as hf:
    print("Available groups in the file:")
    groups = list(hf.keys())      # All high-voltage measurement groups (e.g., "1500(1)")
    for i, group in enumerate(groups):
        print(f"{i + 1}. {group}")

    # ------------------------------------------------------------------
    # To select which measurement group to load
    # ------------------------------------------------------------------
    while True:
        try:
            group_index = int(input("\nSelect a group by entering its number: ")) - 1
            if 0 <= group_index < len(groups):
                selected_group = groups[group_index]
                break
            else:
                print("Invalid number. Please select a valid group number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # ------------------------------------------------------------------
    # To load waveforms and metadata from the selected group
    # ------------------------------------------------------------------

    y_values = hf[f"{selected_group}/waveforms/data"][:]

    # Experimental metadata needed to reconstruct voltages from ADC values
    samplerate = hf[f"{selected_group}/waveform_info/samplerate"][()]
    YOFF = hf[f"{selected_group}/waveform_info/y_off"][:]        # Vertical offset
    YMU = hf[f"{selected_group}/waveform_info/v_gain"][:]       # Vertical gain (ADC → volts)
    Measurement_time = hf[f"{selected_group}/waveform_info/measurement_time"][()]
    # Saved timestamp is stored as bytes, so decoding to a string here:
    timestamp = hf[f"{selected_group}/waveform_info/timestamp"][()].decode('utf-8')

    # Print basic metadata for verification
    print(f"\nLoaded data from group: {selected_group}")
    print("y_values shape:", y_values.shape)
    print("Measurement_time:", Measurement_time)
    print("YOFF:", YOFF)
    print("YMU:", YMU)
    print("Samplerate:", samplerate)
    print("Timestamp:", timestamp)

# ----------------------------------------------------------------------
# Reconstruct time axis based on number of samples and measurement time
# ----------------------------------------------------------------------
time_axis = np.linspace(0, Measurement_time, y_values.shape[2])

# ----------------------------------------------------------------------
# Plot the loaded waveforms
# ----------------------------------------------------------------------
fig, ax = plt.subplots()
for waveform in y_values[0]:
    ax.plot(time_axis, waveform * YMU[0] + YOFF[0])

# Title includes PMT SN and HV (extracted from the group name)
ax.set_title(f'Recorded pulses - SN: TM0007, HV: {selected_group.split("(")[0]} V', fontsize=30)
ax.set_xlabel('Time (s)', fontsize=30)
ax.set_ylabel('Amplitude (V)', fontsize=30)

# Show timestamp in a readable box inside the plot. Optional block, can be commented out:
ax.text(0.05, 0.95, f"Timestamp: {timestamp}", transform=ax.transAxes, fontsize=14,
        bbox=dict(facecolor='white', alpha=0.5))

plt.show()
