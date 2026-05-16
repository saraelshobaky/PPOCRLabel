import json

def scale_coordinates(input_path, output_path, current_scale=0.6):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            # The file is separated by a tab: [filename] \t [JSON string]
            try:
                filename, json_string = line.split('\t', 1)
            except ValueError:
                print(f"Skipping malformed line: {line[:50]}...")
                continue
                
            # Parse the JSON array
            data = json.loads(json_string)
            
            # Iterate over every bounding box entry
            for item in data:
                if 'points' in item:
                    new_points = []
                    for point in item['points']:
                        x, y = point
                        
                        # # Scale back from current_scale to 1.0 and round to nearest integer
                        # new_x = int(round(x / current_scale))
                        # new_y = int(round(y / current_scale))

                        # Scale back to from 1.0 to current_scale and round to nearest integer
                        new_x = int(round(x * current_scale))
                        new_y = int(round(y * current_scale))
                        
                        new_points.append([new_x, new_y])
                        
                    # Update the item with the scaled points
                    item['points'] = new_points
                    
            # Convert back to a JSON string without escaping Arabic characters
            updated_json = json.dumps(data, ensure_ascii=False)
            
            # Write the updated line to the new file
            f_out.write(f"{filename}\t{updated_json}\n")

    print(f"Update complete! Scaled coordinates saved to '{output_path}'.")

# Run the function
scale_coordinates('training_data2/pngs/Label.txt', 'training_data2/pngs/Label_scaled.cach', current_scale=0.6)