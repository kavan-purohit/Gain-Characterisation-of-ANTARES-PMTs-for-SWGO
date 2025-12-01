"""

This script is to be used to see individual waveform from any saved .h5 file

"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# --- List available SNs from DataNew folder ---
folder = "DataNew"
files = [f for f in os.listdir(folder) if f.endswith(".h5")]
sns = [f[:-3] for f in files]  # strip '.h5'

if not sns:
    print("No .h5 files found in the folder 'DataNew'.")
    exit()

print("Available Serial Numbers:")
for i, sn in enumerate(sns):
    print(f"{i + 1}. {sn}")

while True:
    try:
        sn_index = int(input("\nSelect a Serial Number by entering its number: ")) - 1
        if 0 <= sn_index < len(sns):
            SN = sns[sn_index]
            break
        else:
            print("Invalid number. Please select a valid SN number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

filename = os.path.join(folder, SN + ".h5")

# --- Select HV group ---
with h5py.File(filename, 'r') as hf:
    groups = list(hf.keys())
    print(f"\nAvailable HV groups in {filename}:")
    for i, group in enumerate(groups):
        print(f"{i + 1}. {group}")

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

threshold = float(input("Enter voltage threshold (e.g., 1 for -1V): "))
threshold = abs(threshold)  # always positive input

group_path = f"{selected_group}/separate_waveform/data"
info_path = f"{selected_group}/waveform_info"

# --- Load data ---
with h5py.File(filename, 'r') as hf:
    raw_waveforms = hf[group_path][:]
    ymu = hf[f"{info_path}/v_gain"][()]
    yoff = hf[f"{info_path}/y_off"][()]
    meas_time = hf[f"{info_path}/measurement_time"][()]
    samplerate = hf[f"{info_path}/samplerate"][()]

# --- Convert to physical units ---
volt_waveforms = raw_waveforms * ymu + yoff
time_axis = np.linspace(0, meas_time, raw_waveforms.shape[1])

# --- Identify indices with negative peaks below threshold ---
peak_vals = np.min(volt_waveforms, axis=1)
indices_above_thresh = np.where(peak_vals < -threshold)[0]

print(f"\nTotal waveforms: {len(volt_waveforms)}")
print(f"Waveforms below -{threshold:.2f} V: {len(indices_above_thresh)}")

# --- Option to plot specific one ---
if len(indices_above_thresh) > 0:
    view = input("Do you want to plot one of them? (y/n): ").strip().lower()
    if view == 'y':
        choice = input("View from (a) all waveforms or (b) only those below threshold? (a/b): ").strip().lower()

        if choice == 'b':
            pool = indices_above_thresh
            print(f"\nAvailable indices (0 to {len(pool)-1}) in filtered set of {len(pool)} waveforms.")
        else:
            pool = np.arange(len(volt_waveforms))
            print(f"\nAvailable indices (0 to {len(pool)-1}) in full set of {len(pool)} waveforms.")

        while True:
            try:
                idx = int(input(f"Enter index to view (0 to {len(pool)-1}): "))
                if 0 <= idx < len(pool):
                    wf_idx = pool[idx]
                    plt.plot(time_axis, volt_waveforms[wf_idx])
                    plt.title(f"Waveform #{wf_idx} | Min = {peak_vals[wf_idx]:.2f} V")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Voltage (V)")
                    plt.grid(True)
                    plt.show()
                else:
                    print("Index out of range.")
            except ValueError:
                break
