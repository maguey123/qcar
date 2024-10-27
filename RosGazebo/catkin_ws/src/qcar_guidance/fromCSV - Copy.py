import csv
from collections import defaultdict

def read_csv_data(filename):
    data = defaultdict(list)
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            color = row[0]
            x, y = float(row[1]), float(row[2])
            data[color].append((x, y))
    return data

def write_formatted_data(data, output_filename):
    with open(output_filename, 'w') as file:
        file.write("x y\n")
        
        # Blue Cone Positions
        file.write("Blue Cone Positions\n")
        for x, y in data['blue']:
            file.write(f"{x:.8e} {y:.8e}\n")
        
        # Yellow Cone Positions
        file.write("Yellow Cone Positions\n")
        for x, y in data['yellow']:
            file.write(f"{x:.8e} {y:.8e}\n")
        
        # Orange Cone Positions
        file.write("Orange Cone Positions\n")
        for x, y in data['big_orange']:
            file.write(f"{x:.8e} {y:.8e}\n")

def main():
    input_filename = 'cone_pos.csv'  # Change this to your input CSV file name
    output_filename = 'output.txt'  # Change this to your desired output file name
    
    data = read_csv_data(input_filename)
    write_formatted_data(data, output_filename)
    print(f"Data has been converted and written to {output_filename}")

if __name__ == "__main__":
    main()