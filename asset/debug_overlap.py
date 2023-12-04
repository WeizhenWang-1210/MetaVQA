# Correcting the column names in the overlap check function
import pandas as pd
def check_overlaps_corrected(data):
    # Group data by id
    grouped_data = data.groupby('id')

    # Iterate through each group and check for overlaps
    overlaps = []
    for id, group in grouped_data:
        # Sort the group by longitude and latitude
        sorted_group = group.sort_values(by=['long', 'lat'])

        # Iterate through the sorted objects to check for overlaps
        for i in range(len(sorted_group) - 1):
            current_obj = sorted_group.iloc[i]
            next_obj = sorted_group.iloc[i + 1]

            # Check if the next object is within the bounds of the current object
            if (next_obj['long'] < current_obj['long'] + current_obj['len']) and \
               (next_obj['lat'] < current_obj['lat'] + current_obj['width']):
                print("!!!!Overlaps Founded")
                print(current_obj)
                print(next_obj)

csv_file_path = 'D:\\research\\metavqa_main\\MetaVQA\\asset\\spawned_objects_log.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)

# Check for overlaps
overlaps = check_overlaps_corrected(data)
