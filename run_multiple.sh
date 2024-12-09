#!/bin/bash

# Directory to loop through
DIR="./configs/"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
  echo "Directory $DIR does not exist."
  exit 1
fi

# Loop through all files in the directory
for file in "$DIR"/*; do
  if [ -f "$file" ]; then
    # Perform an operation on the file
    echo "Processing file: $file"
    python main.py --config "$file"
  fi
done