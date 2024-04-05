import os

# Define the paths to the main folders
main_folders = ['Dirty','Structured','Textual']

# Loop through each main folder
for main_folder in main_folders:
    main_folder_path = os.path.join('Datasets', main_folder)
    
    # List all subfolders in the main folder
    subfolders = [f.name for f in os.scandir(main_folder_path) if f.is_dir()]
    
    # Loop through each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder_path, subfolder)
        
        # List all CSV files in the subfolder
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
