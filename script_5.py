"""
======================================================================
    Script to be used just to check all available keys in any group
    created or edited from script_1.py or script_3.1.py

    don't forget to change the name of your desired .h5 file!
======================================================================
"""

import h5py

h5_file = "DataNew/TM0007_2.h5"


def explore_h5_file(h5_filename):
    with h5py.File(h5_filename, 'r') as hf:
        # Display all groups
        groups = list(hf.keys())
        if not groups:
            print("No groups found in the HDF5 file.")
            return

        print("\nAvailable groups in the file:")
        for i, group in enumerate(groups):
            print(f"{i + 1}. {group}")

        # Ask the user to select a group
        while True:
            try:
                group_index = int(input("\nEnter the number of the group you want to explore: ")) - 1
                if 0 <= group_index < len(groups):
                    selected_group = groups[group_index]
                    break
                else:
                    print("Invalid number. Please enter a valid group number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Display keys inside the selected group
        print(f"\nKeys in group '{selected_group}':")
        keys = list(hf[selected_group].keys())
        for key in keys:
            print(f"  - {key}")

        # Ask if user wants to explore a key further
        while True:
            explore_more = input("\nDo you want to explore the contents of a key? (y/n): ").strip().lower()
            if explore_more == 'n':
                break
            elif explore_more == 'y':
                print("\nAvailable keys:", keys)
                key_choice = input("Enter the key name to explore: ").strip()
                if key_choice in keys:
                    item = hf[f"{selected_group}/{key_choice}"]
                    if isinstance(item, h5py.Group):
                        print(f"\n'{key_choice}' is a group with the following subkeys:")
                        subkeys = list(item.keys())
                        print(subkeys)

                        # Display values inside each subkey
                        for subkey in subkeys:
                            sub_item = item[subkey]
                            if isinstance(sub_item, h5py.Dataset):
                                print(f"{subkey} = {sub_item[()]}")
                            else:
                                print(f"{subkey} = (Cannot display, complex structure)")
                    elif isinstance(item, h5py.Dataset):
                        print(f"\n'{key_choice}' contains the following data:")
                        print(item[()])  # Print dataset values
                else:
                    print("Invalid key name. Try again.")
            else:
                print("Invalid input. Enter 'y' to explore or 'n' to exit.")


explore_h5_file(h5_file)
