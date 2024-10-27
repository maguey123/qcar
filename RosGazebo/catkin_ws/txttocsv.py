import csv

def convert_to_csv(input_filename, output_filename):
    # Read the input file
    with open(input_filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Remove the first line if it's just a number (seems to be a count)
    if lines[0].isdigit():
        lines = lines[1:]
    
    # Prepare data for CSV
    data = []
    time_seconds = 0.0  # Start at 0 seconds
    time_increment = 0.01  # 10ms = 0.01 seconds
    
    for line in lines:
        try:
            # Split the fixed-width format at position 7
            input_val = float(line[:7])
            current_val = float(line[7:])
            
            data.append({
                'time_s': f"{time_seconds:.2f}",
                'current': current_val,
                'input': input_val,
                'motortorque': 0  # Placeholder for motor torque calculation
            })
            time_seconds += time_increment
            
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid line: {line}, Error: {e}")
            continue

    # Write to CSV
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['time_s', 'current', 'input', 'motortorque']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
        
    return len(data)

# Convert and save to CSV
input_file = "CoolTerm Capture 2024-10-25 14-30-12.txt"
num_records = convert_to_csv(input_file, 'motor_measurements.csv')
print(f"Successfully converted {num_records} measurements to motor_measurements.csv")