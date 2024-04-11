import csv
import re

# Replace 'your_file.csv' with the path to your actual CSV file
file_path = '/home/das160/sparse-register-tiling/results/bench_executor_16thrds.csv'

def extract_num_dense(matrix_string):
    # Define the regular expression pattern to match the 'num_dense' part of the string
    # The pattern below assumes 'num_dense' is a series of digits (\d+)
    # and captures it after the fifth underscore and before the sixth underscore followed by a digit or the word 'uniform' or 'nonuniform'
    pattern = r'Matrix_\d+_\d+_\d+_\d+_(\d+)_\d+_(uniform|nonuniform)'
    
    # Use re.search to find a match for the pattern
    match = re.search(pattern, matrix_string)
    
    # If a match is found, return the captured group (num_dense)
    if match:
        return int(match.group(1))
    else:
        raise Exception(f"No match found for {matrix_string}")

# Open and read the CSV file
with open(file_path, mode='r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Extract the matrix name and the numbers
        matrix_name = row[0]
        if matrix_name.endswith("_uniform"):
            continue
        if matrix_name.endswith("_10_nonuniform"):
            continue
        if matrix_name.endswith("30_nonuniform"):
            continue
        if matrix_name.endswith("40_nonuniform"):
            continue
        if matrix_name.endswith("75_nonuniform"):
            continue
        if matrix_name.endswith("99_nonuniform"):
            continue
        num_dense = extract_num_dense(matrix_name)
        if num_dense % 50 != 0:
            continue
        numbers = list(map(int, row[1:]))
        
        # Calculate the average of the numbers and divide by 1000
        average = sum(numbers) / len(numbers) / 1000
        
        # Print the result
        print(f"{matrix_name}: {average:.2f}")
