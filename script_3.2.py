"""
=============================================================================
This is a small script around:

    data.analysis_complete_data(...)

Meant for Running a full, standardized analysis pass on any existing HDF5 file

This script does NOT modify any of analysis logic; it only forwards
the filename and options to modules.data_analysis.analysis_complete_data.
=============================================================================
"""
import argparse
import sys

# Try to import the local analysis module
# Note: This assumes you run the script from the project root so that
#       the "modules" package is on the Python path.
try:
    import modules.data_analysis as data
except ImportError as e:
    print(
        "ERROR: Could not import 'modules.data_analysis'. "
        "Run this from the project root so Python can find the 'modules' package.",
        file=sys.stderr,
    )
    raise


def main():
    # Parse command-line arguments and dispatch to data.analysis_complete_data.

    parser = argparse.ArgumentParser(
        description=(
            "Analyze complete dataset already saved to HDF5 "
            "(wrapper around data.analysis_complete_data)"
        )
    )
    # Positional argument: path to the existing HDF5 file
    parser.add_argument("h5_path", help="Path to the HDF5 file you want to analyze")
    args = parser.parse_args()

    filename = args.h5_path

    # Core call: this is where the full analysis is executed.
    # - reanalyse=False     → reuse existing intermediate results where possible
    # - saveresults=False   → don't overwrite/store new results (pure inspection)
    # - log=True            → print log output to the console
    # - plot='plot'         → produce plots (disable/change as needed)
    # - nominal_gains=[5e7] → list of nominal gain values used as reference

    # data.analysis_complete_data(filename, reanalyse=False, saveresults=False, log=True, plot='plot')
    data.analysis_complete_data(
        filename,
        reanalyse=False,
        saveresults=False,
        log=True,
        plot='plot',
        nominal_gains=[5e7],
    )


if __name__ == "__main__":
    # Standard Python entry point pattern
    main()
