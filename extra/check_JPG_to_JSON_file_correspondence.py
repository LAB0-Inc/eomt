#!/usr/bin/env python3
"""
Check one-to-one correspondence between JPG images and JSON files.
Lists files that don't have a matching counterpart.
"""

import os
import sys
from pathlib import Path

def check_correspondence(jpg_folder, json_folder):
    """
    Check for one-to-one correspondence between JPG and JSON files.

    Args:
        jpg_folder: Path to folder containing JPG images
        json_folder: Path to folder containing JSON files
    """
    # Get all JPG files (case-insensitive)
    jpg_files = set()
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
        jpg_files.update(Path(jpg_folder).glob(ext))
    jpg_names = {f.stem for f in jpg_files}

    # Get all JSON files
    json_files = set(Path(json_folder).glob('*.json'))
    json_names = {f.stem for f in json_files}

    # Find mismatches
    jpg_without_json = jpg_names - json_names
    json_without_jpg = json_names - jpg_names

    # Print results
    print(f"JPG folder: {jpg_folder}")
    print(f"JSON folder: {json_folder}")
    print(f"\nTotal JPG files: {len(jpg_names)}")
    print(f"Total JSON files: {len(json_names)}")
    print(f"Matching pairs: {len(jpg_names & json_names)}")

    if jpg_without_json:
        print(f"\n⚠ JPG files without corresponding JSON ({len(jpg_without_json)}):")
        for name in sorted(jpg_without_json):
            print(f"  - {name}.jpg")

    if json_without_jpg:
        print(f"\n⚠ JSON files without corresponding JPG ({len(json_without_jpg)}):")
        for name in sorted(json_without_jpg):
            print(f"  - {name}.json")

    if not jpg_without_json and not json_without_jpg:
        print("\n✓ Perfect correspondence! All files have matching pairs.")
        return 0
    else:
        return 1

if __name__ == "__main__":
    jpg_folder = '/workspace/data/Datasets/Existing/training_images'
    json_folder = '/workspace/data/Datasets/Existing/training_annotations_labelme_format'

    # Check if folders exist
    if not os.path.isdir(jpg_folder):
        print(f"Error: JPG folder '{jpg_folder}' does not exist")
        sys.exit(1)

    if not os.path.isdir(json_folder):
        print(f"Error: JSON folder '{json_folder}' does not exist")
        sys.exit(1)

    exit_code = check_correspondence(jpg_folder, json_folder)
    sys.exit(exit_code)
