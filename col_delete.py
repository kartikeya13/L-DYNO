def delete_columns(input_file, output_file, columns_to_delete):
    with open(input_file, 'r') as file_in, open(output_file, 'w') as file_out:
        for line in file_in:
            # Split the line into columns
            columns = line.strip().split('\t')  # Assuming tab-separated columns, adjust delimiter as needed

            # Remove the specified columns
            remaining_columns = [col for idx, col in enumerate(columns) if idx + 1 not in columns_to_delete]

            # Write the modified line to the output file
            modified_line = '\t'.join(remaining_columns) + '\n'  # Join columns with tab delimiter
            file_out.write(modified_line)

# Example usage
input_file = 'input.txt'  # Replace with the path to your input file
output_file = 'output.txt'  # Replace with the desired path for the output file
columns_to_delete = [2, 4]  # Specify the column numbers you want to delete

delete_columns(input_file, output_file, columns_to_delete)

