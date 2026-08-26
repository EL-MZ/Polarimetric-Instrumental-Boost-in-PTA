#!/bin/bash

# Loop over all files matching the J*.tim pattern
for tim_file in J*.tim; do
    # Ensure the file exists (prevents errors if no files match the pattern)
    [ -e "$tim_file" ] || continue

    # Extract the pulsar name (removes the .tim extension)
    pulsar="${tim_file%.tim}"
    echo "--------------------------------------------------"
    echo "Processing pulsar: $pulsar"
    
    # Run psrcat to get RAJ and DECJ. 
    # awk '/^[0-9]+/ {print $2, $5}' finds the data row starting with a number 
    # and extracts the 2nd (RAJ) and 5th (DECJ) columns.
    coords=$(psrcat -c "RAJ DECJ" "$pulsar" | awk '/^[0-9]+/ {print $2, $5}')
    
    if [ -z "$coords" ]; then
        echo "Error: Could not retrieve coordinates for $pulsar from psrcat. Skipping."
        continue
    fi
    
    # Assign RAJ and DECJ to separate variables
    read -r RAJ DECJ <<< "$coords"
    
    # Combine them for getHorizon (e.g., 07:11:54.180042-68:30:47.36454)
    # Note: psrcat normally outputs DEC with its sign (+ or -), so direct concatenation works.
    coord_string="${RAJ}${DECJ}"
    echo "Extracted Coordinates: $coord_string"
    
    # Define and clear the output .ang file
    ang_file="${pulsar}.ang"
    > "$ang_file"
    
    echo "Calculating parallactic angles..."
    
    # Read the 3rd column (MJD) from the .tim file
    awk '{print $3}' "$tim_file" | while read -r mjd; do
        
        # Safety check: Ensure the value is actually a number (ignores TEMPO/TEMPO2 header lines like 'FORMAT 1')
        if [[ ! "$mjd" =~ ^[0-9]+\.[0-9]+$ ]]; then
            continue
        fi
        
        # Run getHorizon and extract the Parallactic angle
        # awk '/^Parallactic:/ {print $2}' isolates the numerical value
        par_angle=$(getHorizon -c "$coord_string" -m "$mjd" -t pks | awk '/^Parallactic:/ {print $2}')
        
        # Append the result to the .ang file if an angle was found
        if [ -n "$par_angle" ]; then
            echo "$mjd $par_angle" >> "$ang_file"
        else
            echo "Warning: Could not compute angle for MJD $mjd"
        fi
        
    done
    
    echo "Finished $pulsar. Saved to $ang_file"
done

echo "--------------------------------------------------"
echo "All processing complete."