"""
====================================================================================

     This script performs PMT charge spectrum analysis, including:

       1. Selecting a PMT serial number and HV group from saved HDF5 waveform files
       2. Loading waveforms and converting ADC samples to physical voltages
       3. Plotting averaged waveforms and defining integration windows
       4. Building the charge histogram from integrated waveforms
       5. Detecting the valley between pedestal and SPE peaks
          (auto mode + optional manual hint mode)
       6. Fitting the full PMT response function using ChargeHistFitter()
       7. Extracting gain, nphe, SPE charge, pedestal parameters, etc.
       8. Automatically saving all fit results back into the same HDF5 file
       9. Generating multiple plots:

- Relies on: modules.data_analysis, ChargeHistFitter, and your HDF5 file structure.

====================================================================================
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import modules.data_analysis as data
import os


# Enable / disable interactive manual valley selection during debugging.
# If set True → the script repeatedly plots the histogram and asks to adjust the valley.

DEBUG_MANUAL_VALLEY = False

# ==============================================================================
# Valley detection function
# This function tries to find the separation between pedestal and SPE peaks.
# ==============================================================================
def detect_valley(hi, bin_edges, window=5, manual_hint=None):
    # Smooth histogram to remove fluctuations. set the value of window to change smoothing scale
    avg_sig = data.moving_average(hi, window)

    # If user suggests an approximate valley location → search nearby
    if manual_hint is not None:
        closest_idx = (np.abs(bin_edges - manual_hint)).argmin()
        search_radius = 10
        start_idx = max(0, closest_idx - search_radius)
        end_idx = min(len(avg_sig), closest_idx + search_radius)
        search_slice = avg_sig[start_idx:end_idx]
        try:
            offset = np.argmin(search_slice)  # local minimum
            valley_index = start_idx + offset
        except ValueError:
            valley_index = closest_idx

    else:
        # Automatic valley detection:
        #   1. find peak (maximum of smoothed histogram)
        #   2. search for a minimum to the right (first ~1/3 of the remaining range)
        l = np.argmax(avg_sig)
        try:
            right_slice = avg_sig[l:]
            offset = np.argmin(right_slice[:int(len(right_slice) / 3)])
            valley_index = l + offset
        except (IndexError, ValueError):
            valley_index = l

    return bin_edges[valley_index]


# ==============================================================================
# Load available HDF5 files as a list, from (here) DataNew folder
# ==============================================================================
folder = "DataNew"
files = [f for f in os.listdir(folder) if f.endswith(".h5")]
sns = [f[:-3] for f in files]  # strip ".h5" → get serial number strings

if not sns:
    print("No .h5 files found in the folder 'DataNew'.")
    exit()

print("Available Serial Numbers:")
for i, sn in enumerate(sns):
    print(f"{i + 1}. {sn}")

# ----------------------------------------------------------------------
# To pick which PMT serial number to analyse
# ----------------------------------------------------------------------
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

# ==============================================================================
# Load selected HV group from the chosen HDF5 file
# ==============================================================================
with h5py.File(filename, 'r') as hf:
    groups = list(hf.keys())
    print(f"\nAvailable HV groups in {filename}:")
    for i, group in enumerate(groups):
        print(f"{i + 1}. {group}")

    # selects HV group
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

    # Load waveforms + metadata
    y_values = hf[f"{selected_group}/waveforms/data"][:]
    samplerate = hf[f"{selected_group}/waveform_info/samplerate"][()]
    YOFF = hf[f"{selected_group}/waveform_info/y_off"][:]
    YMU = hf[f"{selected_group}/waveform_info/v_gain"][:]
    Measurement_time = hf[f"{selected_group}/waveform_info/measurement_time"][()]
    timestamp = hf[f"{selected_group}/waveform_info/timestamp"][()].decode('utf-8')
    SN = hf[f"{selected_group}/waveform_info/serial_number"][()].decode('utf-8')

# ----------------------------------------------------------------------
# Waveform preparation
# ----------------------------------------------------------------------
y_values = np.squeeze(y_values)  # remove trivial dimensions
y_values = y_values * YMU + YOFF  # convert ADC → Volts

# Plot mean waveform
data.mean_plot(y_values, block=True)

# Default integration windows (pedestal + signal)
# Change these valus as required, this is just a general range scale in my measurements
int_ranges = (0, 220, 250, 450)
data.mean_plot(y_values, block=True, int_ranges=int_ranges)

# To refine integration windows. If you think the int ranges are bit off, you can visually
# see that and change again. NB: this chance to change will only be given once, then it
# will continue with later steps!
if input("\nProceed with these integration ranges? (y/n): ").strip().lower() == 'n':
    ped_min = int(input("Enter new Pedestal minimum index: "))
    ped_max = int(input("Enter new Pedestal maximum index: "))
    sig_min = int(input("Enter new Signal minimum index: "))
    sig_max = int(input("Enter new Signal maximum index: "))
    int_ranges = (ped_min, ped_max, sig_min, sig_max)
    data.mean_plot(y_values, block=True, int_ranges=int_ranges)

# ==============================================================================
# Build charge histogram from the integrated waveform
# ------------------------------------------------------------------------------
hi, bin_edges, int_ranges = data.histogramm(y_values, *int_ranges, plot=True)

# Horizontal scale → convert samples to seconds
h_int = 1 / samplerate


# ==============================================================================
# Valley detection (manual mode OR automatic mode -
# by default I used automatic one, and it is better I guess)
# ==============================================================================
if DEBUG_MANUAL_VALLEY:
    manual_hint = None
    use_manual_hint = False

    # Repeated interactive loop allowing user to inspect/change valley guess
    while True:
        if use_manual_hint:
            valley = detect_valley(hi, bin_edges, manual_hint=manual_hint)
        else:
            valley = detect_valley(hi, bin_edges)

        plt.semilogy(bin_edges, hi, label="Histogram")
        plt.axvline(valley, color='r', linestyle='--', label=f"Detected Valley: {valley:.3f}")
        plt.legend()
        plt.title("Histogram with Detected Valley")
        plt.show(block=False)

        satisfied = input(f"Detected valley at {valley:.3f}. Are you satisfied? (y/n): ").strip().lower()
        plt.close()

        if satisfied == 'y':
            break

        try:
            manual_hint = float(input("Enter your hint for the valley (x-position): "))
            use_manual_hint = True
        except ValueError:
            print("Invalid input. Using automatic valley detection again...")
            use_manual_hint = False

    fitter = data.ChargeHistFitter()
    fitter.pre_fit(bin_edges, hi, valley=valley)

else:
    # Automatic valley detection (used by me)
    fitter = data.ChargeHistFitter()
    fitter.pre_fit(bin_edges, hi)

    # Compute valley estimate for reporting
    valley = bin_edges[np.argmin(hi[np.argmax(hi):np.argmax(hi) + len(hi) // 5]) + np.argmax(hi)]


# ==============================================================================
# Fit the PMT response function (pedestal + SPE + multi-photoelectron model)
# ==============================================================================
fitter.fit_pmt_resp_func(bin_edges, hi, print_level=0, fixed_parameters=["entries"])

# Compute gain:    gain = SPE_charge * (dt/50Ω) / electron_charge
gain = fitter.popt_prf['spe_charge'] * h_int / (50 * data.e)
nphe = fitter.popt_prf['nphe']

# Error propagation for gain
try:
    gain_err = np.sqrt(fitter.pcov_prf[('spe_charge', 'spe_charge')]) * h_int / (50 * data.e)
except:
    gain_err = np.nan

# ------------------------------------------------------------------------------
# Valley-to-Peak ratio (simple diagnostic)
# ------------------------------------------------------------------------------
valley_idx = (np.abs(bin_edges - valley)).argmin()
valley_height = hi[valley_idx]
peak_height = np.max(hi)
vp_ratio = valley_height / peak_height

print(f"\nGain at {selected_group.split('(')[0]}V = {gain:.3e} ± {gain_err:.3e} with {nphe:.2f} photoelectrons")
print(f"Valley-to-Peak Ratio = {vp_ratio:.3f}")


# ==============================================================================
# Save fit results into the same HDF5 file under .../fit_results/
# ==============================================================================
with h5py.File(filename, 'a') as hf:
    if f"{selected_group}/fit_results" not in hf:
        fit_results_group = hf.create_group(f"{selected_group}/fit_results")
    else:
        fit_results_group = hf[f"{selected_group}/fit_results"]

    # Helper: safely overwrite existing datasets
    def safe_write(group, key, value):
        if key in group:
            del group[key]
        group.create_dataset(key, data=value)

    # Extract individual parameter errors from covariance matrix
    prf_errors = {}
    for param in fitter.m.parameters:
        try:
            prf_errors[param] = np.sqrt(fitter.pcov_prf[(param, param)])
        except:
            prf_errors[param] = np.nan

    # Print fitted parameters for terminal output
    for param in ["nphe", "spe_charge", "spe_sigma", "ped_mean", "ped_sigma", "entries"]:
        val = fitter.popt_prf[param]
        err = prf_errors.get(param, np.nan)
        print(f"{param}: {val:.4e} ± {err:.4e}")

    # Save the fit results via helper from data_analysis
    data.add_fit_results(filename, selected_group, gain, nphe, gain_err, int_ranges)

    # Save additional diagnostic parameters
    safe_write(fit_results_group, "valley", valley)
    safe_write(fit_results_group, "valley_to_peak_ratio", vp_ratio)
    safe_write(fit_results_group, "ped_mean", fitter.popt_ped["mean"])
    safe_write(fit_results_group, "ped_sigma", fitter.popt_ped["sigma"])
    safe_write(fit_results_group, "spe_charge", fitter.spe_charge)
    safe_write(fit_results_group, "spe_sigma", fitter.popt_spe["sigma"])
    safe_write(fit_results_group, "entries", fitter.popt_ped["A"] + fitter.popt_spe["A"])

    # Save every PRF parameter with prefix
    for param in fitter.m.parameters:
        safe_write(fit_results_group, f"prf_{param}", fitter.popt_prf[param])
    for param in fitter.m.parameters:
        value = fitter.popt_prf[param]
        print(f"Saving prf_{param} = {fitter.popt_prf[param]:.4e}")


# ==============================================================================
# Several Plot types (1-5).
# ==============================================================================


# ==============================================================================
# 1) Plot results: fit + histogram + valley + annotation block
# ==============================================================================
fit_text = f"""\nGain = {gain:.2e} ± {gain_err:.2e}
Valley = {valley:.1f}
Valley-to-Peak = {vp_ratio:.3f}
nphe = {nphe:.2f} ± {prf_errors['nphe']:.2f}
SPE charge = {fitter.popt_prf['spe_charge']:.2f} ± {prf_errors['spe_charge']:.2f}
SPE sigma = {fitter.popt_prf['spe_sigma']:.2f} ± {prf_errors['spe_sigma']:.2f}
Ped mean = {fitter.popt_prf['ped_mean']:.2f} ± {prf_errors['ped_mean']:.2f}
Ped sigma = {fitter.popt_prf['ped_sigma']:.2f} ± {prf_errors['ped_sigma']:.2f}
Entries = {fitter.popt_prf['entries']:.0f}
"""
# Just removed this command to not print the error for entries
# Entries = {fitter.popt_prf['entries']:.0f} ± {prf_errors['entries']:.0f}


plt.semilogy(bin_edges, hi, label='Histogram')
plt.plot(bin_edges, fitter.opt_prf_values, label='Fit')
plt.axvline(valley, linestyle='--', color='r', label='Valley')
plt.ylim(1e-1, np.max(hi) * 1.2)
plt.title(f"Fit results for SN: {SN} at HV: {selected_group.split('(')[0]} V using Hamamatsu 14-Dy base")
plt.text(0.97, 0.97, fit_text, transform=plt.gca().transAxes,
         va='top', ha='right', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# ==============================================================================
# 2) plot: Charge spectrum in pC
# ------------------------------------------------------------------------------
# Convert histogram x-axis → physical charge in pC
# Q[pC] = (charge_unit * dt / 50Ω) * 1e12
# ==============================================================================
bin_edges_pc = bin_edges * h_int / 50 * 1e12
valley_pc = valley * h_int / 50 * 1e12

plt.figure()
plt.semilogy(bin_edges_pc, hi, label='Histogram')
plt.plot(bin_edges_pc, fitter.opt_prf_values, label='Fit')
plt.axvline(valley_pc, linestyle='--', color='r', label='Valley')
plt.ylim(1e-1, np.max(hi) * 1.2)
plt.title(f"Fit results for SN: {SN} at HV: {selected_group.split('(')[0]} V (Charge in pC)")
plt.xlabel("Charge (pC)")
plt.ylabel("Counts")
plt.text(0.97, 0.97, fit_text, transform=plt.gca().transAxes,
         va='top', ha='right', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


###################################################################################
# 3) Peak-to-Valley using model (robust P/V), and without valley line
###################################################################################
# model_y = np.asarray(fitter.opt_prf_values)
# obs_y = np.asarray(hi)
#
# # Valley index from model-based valley location
# valley_idx = int(np.argmin(np.abs(bin_edges - valley)))
# valley_idx = np.clip(valley_idx, 0, len(model_y) - 1)
#
# # Find peak (use model → avoids noise)
# search_mask = np.arange(len(model_y)) > valley_idx
# if np.any(search_mask):
#     peak_rel_idx = np.argmax(model_y[search_mask])
#     peak_idx = np.flatnonzero(search_mask)[peak_rel_idx]
# else:
#     peak_idx = int(np.argmax(model_y))
#
# valley_height = float(model_y[valley_idx])
# peak_height = float(model_y[peak_idx])
# p2v_ratio = (peak_height / valley_height) if valley_height > 0 else np.inf
#
# p2v_str = "∞" if not np.isfinite(p2v_ratio) else f"{p2v_ratio:.2f}"
#
# fit_text_pv = (
#     f"\nGain = {gain:.2e} ± {gain_err:.2e}"
#     f"\nValley = {valley:.1f}"
#     f"\nPeak-to-Valley = {p2v_str}"
#     f"\nnphe = {nphe:.2f} ± {prf_errors.get('nphe', np.nan):.2f}"
#     f"\nSPE charge = {fitter.popt_prf['spe_charge']:.2f} ± {prf_errors.get('spe_charge', np.nan):.2f}"
#     f"\nSPE sigma = {fitter.popt_prf['spe_sigma']:.2f} ± {prf_errors.get('spe_sigma', np.nan):.2f}"
#     f"\nPed mean = {fitter.popt_prf['ped_mean']:.2f} ± {prf_errors.get('ped_mean', np.nan):.2f}"
#     f"\nPed sigma = {fitter.popt_prf['ped_sigma']:.2f} ± {prf_errors.get('ped_sigma', np.nan):.2f}"
#     f"\nEntries = {fitter.popt_prf['entries']:.0f}"
# )
#
# plt.figure()
# plt.semilogy(bin_edges, hi, label='Histogram')
# plt.plot(bin_edges, model_y, label='Fit')
# plt.ylim(1e-1, np.max(hi) * 1.2)
# plt.title(f"Fit results for SN: {SN} at HV: {selected_group.split('(')[0]} V")
# plt.xlabel("Charge (arb. units)")
# plt.ylabel("Counts")
# plt.text(0.97, 0.97, fit_text_pv, transform=plt.gca().transAxes,
#          va='top', ha='right', fontsize=9,
#          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.show()


# ==============================================================================
# 4) Minimal plot: histogram only (log-y)
# ==============================================================================
# plt.figure()
# plt.semilogy(bin_edges, hi)
# plt.ylim(1e0, np.max(hi) * 1.2)
# plt.title(f"Fit results for SN: {SN} at HV: {selected_group.split('(')[0]} V")
# plt.xlabel("Charge (arb. units)")
# plt.ylabel("Counts")
# plt.show()


# ==============================================================================
# 5) single-PE charge spectrum (in pC)
# ==============================================================================
# try:
#     x_pc = bin_edges_pc
# except NameError:
#     x_pc = None
#
# # Attempt fallbacks for pC axes
# for _cand in ("bin_edges_pC", "charge_bin_edges_pc", "x_pC"):
#     if x_pc is None and _cand in globals():
#         x_pc = globals()[_cand]
#
# if x_pc is None:
#     if hasattr(fitter, "to_pC"):
#         x_pc = fitter.to_pC(np.asarray(bin_edges))
#     elif hasattr(fitter, "q_to_pC"):
#         x_pc = np.asarray(bin_edges) * float(fitter.q_to_pC)
#     else:
#         raise RuntimeError("Cannot find pC x-axis. Please define bin_edges_pc.")
#
# plt.figure()
# plt.semilogy(x_pc, hi)
# plt.ylim(1e0, np.max(hi) * 1.2)
# plt.xlabel("Charge (pC)")
# plt.ylabel("Counts")
# plt.title(f"SPE charge spectrum (including pedestal), HV = {selected_group.split('(')[0]} V")
# plt.tight_layout()
# plt.show()

###################################################################################

# for calculations of nominal high voltages
# (now transferred this whole segment as a new script: script_3.2):

# data.analysis_complete_data(filename, reanalyse=False, saveresults=False, log=True, plot='plot', nominal_gains=[5e7])
# data.analysis_complete_data(filename, reanalyse=False, saveresults=False, log=True, plot='plot')
