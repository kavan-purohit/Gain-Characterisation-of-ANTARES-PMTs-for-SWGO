"""

    Script to be used only to delete a group from the h5 file created by script_1.py

    don't forget to change the name of your desired .h5 file!

"""

import h5py

filename = "DataNew/TM0007.h5"

def list_groups(h5_file):
    with h5py.File(h5_file, "r") as hf:
        groups = list(hf.keys())
        print("\nAvailable groups:")
        for i, group in enumerate(groups):
            print(f"{i + 1}. {group}")
        return groups

def delete_group(h5_file, group_name):
    with h5py.File(h5_file, "a") as hf:
        if group_name in hf:
            del hf[group_name]
            print(f"Group '{group_name}' has been deleted.")
        else:
            print(f"Error: Group '{group_name}' not found in the file.")

if __name__ == "__main__":
    # List available groups
    available_groups = list_groups(filename)

    # Ask user for the group to delete
    group_to_delete = input("\nEnter the group name to delete {don't enter the group number, enter the group NAME - eg: 1200(1)}: ").strip()

    # Confirm deletion
    confirm = input(f"Are you sure you want to delete '{group_to_delete}'? (y/n): ").strip().lower()
    if confirm == "y":
        delete_group(filename, group_to_delete)
    else:
        print("Operation canceled.")
