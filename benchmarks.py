import os
import torch
import numpy as np

def find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, target_id):
    # Build the search pattern based on inputs
    stacks_str = f"{n_rows:02d}"  # e.g., 16 stacks -> "16"
    tiers_str = f"{n_tiers:02d}"  # e.g., 6 tiers -> "06"
    id_str = f"{target_id:03d}"  # e.g., id 3 -> "003"

    if inst_type == "random":
        folder_path += "/individual, random"
        bays_str = f"R{n_bays:02d}"  # e.g., 2 bays -> "R02"
    elif inst_type == "upsidedown":
        folder_path += "/individual, upside down"
        bays_str = f"U{n_bays:02d}"  # e.g., 2 bays -> "U02"
    else:
        raise ValueError(f"No instance type of '{inst_type}'")


    # Look for the matching file
    matching_files = [
        f for f in os.listdir(folder_path) 
        if f.startswith(f"{bays_str}{stacks_str}{tiers_str}") and f.endswith(f"_{id_str}.txt")
    ]
    
    if not matching_files:
        raise FileNotFoundError(f"No file found matching the criteria: bays={n_bays}, stacks={n_rows}, tiers={n_tiers}, id={target_id}")
    elif len(matching_files) > 1:
        raise ValueError(f"There are more than one file: {matching_files}")
    
    # Assuming the first match is the target file
    target_file = matching_files[0]
    print(f"Processing file: {target_file}")
    
    file_path = os.path.join(folder_path, target_file)
    return parse_container_file(file_path, n_bays, n_rows, n_tiers), target_file

def parse_container_file(file_path, n_bays, n_stacks, n_tiers):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Create a zero-filled numpy array of shape (n_bays * n_stacks, n_tiers)
    container_matrix = np.zeros((n_bays * n_stacks, n_tiers), dtype=int)
    
    # Process each subsequent line
    for line in lines[1:]:  # Skip the first line
        values = list(map(int, line.split()))
        bay, stack, num_tiers = values[:3]
        container_numbers = values[3:]
        
        # Remove duplicates (since IDs are repeated twice)
        unique_container_numbers = list(dict.fromkeys(container_numbers))  # Preserve order while removing duplicates
        
        if len(unique_container_numbers) != num_tiers:
            ValueError(f'len(unique_container_numbers)(={len(unique_container_numbers)})'
                       +f' != numtiers(={num_tiers})')
        if len(unique_container_numbers) > n_tiers:
            ValueError(f'len(unique_container_numbers)(={len(unique_container_numbers)})'
                       +f' > n_tiers(={n_tiers})')
        
        # Pad with zeros if the number of unique containers is less than max tiers
        padded_containers = unique_container_numbers + [0] * (n_tiers - len(unique_container_numbers))
        
        # Compute the index in the matrix
        stack_index = (bay - 1) * n_stacks + (stack - 1)
        
        # Fill the matrix with container numbers (bottom-up stacking)
        container_matrix[stack_index] = padded_containers
    
    # Convert to torch tensor with shape (1, n_bays * n_stacks, n_tiers)
    container_tensor = torch.tensor(container_matrix).unsqueeze(0).float()
    return container_tensor


if __name__ == '__main__':
    # Example usage
    folder_path = "./Lee_instances"  # Replace with the folder containing your files
    inst_type = "random"
    n_bays = 2
    n_rows = 16
    n_tiers = 6
    id = 3

    try:
        container_tensor = find_and_process_file(folder_path, inst_type, n_bays, n_rows, n_tiers, id)
        print(container_tensor.shape)  # Should be (1, n_bays * n_stacks, n_tiers)
        print(container_tensor)
    except FileNotFoundError as e:
        print(e)
