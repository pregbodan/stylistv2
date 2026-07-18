"""
ingest_external_datasets.py
----------------------------
Inspects the five external dataset FOLDERS (once downloaded and placed under
data/external/) and maps anything usable into the project's two training
pipelines:

  1. TEXT  -> data/hardware_faults_dataset.csv (the {text, fault_category}
     schema train_model.py uses for the TF-IDF/SVM classifier)
  2. IMAGE -> a labeled image manifest the vision/Gemini-assisted pipeline
     can use as few-shot reference examples or for future fine-tuning
     (data/external_image_manifest.csv)

Each dataset is a FOLDER, not a single file, and may contain:
  - one or more CSVs at any depth (some datasets ship multiple CSVs, e.g.
    a main table plus a lookup/category table)
  - one or more image subfolders (e.g. class-labeled directories of photos)
  - both at once

Expected folder layout under data/external/:
    data/external/computer_hardware/        <- General Computer Hardware Dataset (Dilshaan Sandhu)
    data/external/electronic_components/    <- Electronic components and devices (aryaminus)
    data/external/electronic_parts/         <- Electronic Parts Dataset (olavomendes)
    data/external/software_reliability/     <- Software Reliability Dataset (vasanthkumarch)
    data/external/software_defects/         <- Software Defect Prediction (semustafacevik)

Each folder's *actual* internal structure is unknown until inspected — this
script does not assume specific filenames inside the folder. It walks each
folder recursively, finds every .csv, and finds every subdirectory that looks
like an image class folder (a directory containing image files directly).

WHY THIS IS A SEPARATE, MANUAL STEP:
This sandbox's outbound network is restricted to a fixed allowlist that does
not include kaggle.com, so these folders cannot be downloaded directly here.
Download them locally, extract them, and place each dataset's folder under
data/external/ using the names above, then run this script.

WHAT THIS SCRIPT DOES, PER CSV FOUND:
  1. Loads it and prints its actual columns + a sample row, so the real
     schema is visible instead of assumed.
  2. Looks for a column that plausibly holds free-text (description, notes,
     issue, fault, etc.). If found, scans each row's text for hardware-
     category keywords and maps matches to the project's existing fault
     categories.
  3. If no usable text column exists (e.g. the CSV is a specs/metrics table
     of model numbers, prices, complexity scores), it is skipped with an
     explanation rather than forcing a fake label.

WHAT THIS SCRIPT DOES, PER IMAGE SUBFOLDER FOUND:
  1. Counts image files and records the folder name as a candidate class
     label, writing folder_name -> file_path rows to
     data/external_image_manifest.csv.
  2. Does NOT move, copy, or upload images anywhere — only catalogs paths,
     since image files can be large and the manifest is enough for a
     downstream step to load them on demand.

NOTHING here is merged into the live training set automatically. All output
goes to *_audit.csv / manifest files for review first — see the printed
summary at the end of the run for next steps.
"""

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = os.path.join(BASE_DIR, "data", "external")
MAIN_DATASET = os.path.join(BASE_DIR, "data", "hardware_faults_dataset.csv")
TEXT_AUDIT_LOG = os.path.join(BASE_DIR, "data", "hardware_faults_dataset_audit.csv")
IMAGE_MANIFEST = os.path.join(BASE_DIR, "data", "external_image_manifest.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

EXPECTED_FOLDERS = {
    "computer_hardware": "General Computer Hardware Dataset (Dilshaan Sandhu)",
    "electronic_components": "Electronic components and devices (aryaminus)",
    "electronic_parts": "Electronic Parts Dataset (olavomendes)",
    "software_reliability": "Software Reliability Dataset (vasanthkumarch)",
    "software_defects": "Software Defect Prediction (semustafacevik)",
}

# Component/keyword -> our fault_category, used to map a row's free-text
# description onto an existing category, IF a usable text column is found.
CATEGORY_HINTS = {
    "power_supply_failure": ["psu", "power supply", "smps"],
    "overheating": ["thermal", "cooling", "fan", "heatsink", "temperature"],
    "ram_failure": ["ram", "memory", "dimm"],
    "storage_failure": ["hdd", "ssd", "hard drive", "hard disk", "storage"],
    "display_gpu_failure": ["gpu", "graphics", "monitor", "display", "video card"],
    "peripheral_issue": ["keyboard", "mouse", "usb", "peripheral", "printer"],
}

TEXT_COLUMN_CANDIDATES = [
    "description", "desc", "comment", "comments", "notes", "issue",
    "fault", "defect", "summary", "title", "name", "component",
]

os.makedirs(EXTERNAL_DIR, exist_ok=True)


def guess_category_from_text(text: str):
    text_lower = str(text).lower()
    for category, hints in CATEGORY_HINTS.items():
        if any(h in text_lower for h in hints):
            return category
    return None


def find_csv_files(folder_path):
    """Recursively finds every .csv under folder_path."""
    csv_paths = []
    for root, _dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, fname))
    return sorted(csv_paths)


def find_image_class_folders(folder_path):
    """
    Finds subdirectories that directly contain image files (a common layout
    for image-classification datasets: class_name/img1.jpg, img2.jpg, ...).
    Returns a dict of {folder_path: [image_file_paths]}.
    """
    class_folders = {}
    for root, _dirs, files in os.walk(folder_path):
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        if image_files:
            class_folders[root] = [os.path.join(root, f) for f in image_files]
    return class_folders


def inspect_and_map_csv(filepath, dataset_label):
    rel_path = os.path.relpath(filepath, EXTERNAL_DIR)
    print(f"\n  --- CSV: {rel_path} ---")

    try:
        df = pd.read_csv(filepath, nrows=2000)
    except Exception as e:
        print(f"    Could not read file: {e}")
        return []

    print(f"    Columns: {list(df.columns)}")
    print(f"    Rows (sampled up to 2000): {len(df)}")
    if len(df) > 0:
        print(f"    Sample row: {df.iloc[0].to_dict()}")

    cols_lower = {c.lower(): c for c in df.columns}
    text_col = next((cols_lower[c] for c in TEXT_COLUMN_CANDIDATES if c in cols_lower), None)

    if text_col is None:
        print("    -> SKIPPED: no recognizable free-text column "
              "(looks like a specs/metrics table, not symptom narratives).")
        return []

    mapped_rows = []
    for _, row in df.iterrows():
        text_value = row.get(text_col)
        if pd.isna(text_value) or not str(text_value).strip():
            continue
        category = guess_category_from_text(text_value)
        if category:
            mapped_rows.append({
                "text": str(text_value).strip(),
                "fault_category": category,
                "source_dataset": dataset_label,
                "source_file": rel_path,
            })

    print(f"    -> Mapped {len(mapped_rows)} row(s) via keyword match on column '{text_col}'.")
    return mapped_rows


def process_dataset_folder(folder_name, label):
    folder_path = os.path.join(EXTERNAL_DIR, folder_name)

    print(f"\n{'=' * 65}")
    print(f"DATASET FOLDER: {folder_name}  ({label})")
    print("=" * 65)

    if not os.path.isdir(folder_path):
        print(f"  [missing] expected folder not found at {folder_path}")
        return [], {}

    csv_files = find_csv_files(folder_path)
    image_class_folders = find_image_class_folders(folder_path)

    if not csv_files and not image_class_folders:
        print("  Folder exists but no CSVs or images were found inside it. "
              "Check the extraction — contents may be nested differently than expected.")
        return [], {}

    print(f"  Found {len(csv_files)} CSV file(s), "
          f"{len(image_class_folders)} folder(s) containing images.")

    mapped_text_rows = []
    for csv_path in csv_files:
        mapped_text_rows.extend(inspect_and_map_csv(csv_path, label))

    image_rows = []
    for class_folder, image_paths in image_class_folders.items():
        class_label = os.path.basename(class_folder.rstrip(os.sep))
        print(f"\n  --- Image folder: {os.path.relpath(class_folder, EXTERNAL_DIR)} "
              f"({len(image_paths)} image(s), candidate label: '{class_label}') ---")
        for p in image_paths:
            image_rows.append({
                "image_path": p,
                "folder_label": class_label,
                "source_dataset": label,
            })

    return mapped_text_rows, image_rows


def main():
    all_text_rows = []
    all_image_rows = []
    any_folder_found = False

    for folder_name, label in EXPECTED_FOLDERS.items():
        text_rows, image_rows = process_dataset_folder(folder_name, label)
        if text_rows or image_rows or os.path.isdir(os.path.join(EXTERNAL_DIR, folder_name)):
            any_folder_found = True
        all_text_rows.extend(text_rows)
        all_image_rows.extend(image_rows)

    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print("=" * 65)

    if not any_folder_found:
        print(f"No expected dataset folders found under {EXTERNAL_DIR}.")
        print("Extract each downloaded dataset into its own folder there, using these names:")
        for folder_name in EXPECTED_FOLDERS:
            print(f"  data/external/{folder_name}/")
        return

    if all_text_rows:
        new_df = pd.DataFrame(all_text_rows)
        new_df.to_csv(TEXT_AUDIT_LOG, index=False)
        print(f"Text:  wrote {len(new_df)} candidate row(s) to "
              f"{os.path.relpath(TEXT_AUDIT_LOG, BASE_DIR)} for manual review.")
        print("       Review it, remove anything wrong, then append good rows to")
        print(f"       {os.path.relpath(MAIN_DATASET, BASE_DIR)} before re-running train_model.py.")
    else:
        print("Text:  no rows mapped from any CSV — likely because the source CSVs are "
              "specs/metrics tables rather than fault-description text. See per-file notes above.")

    if all_image_rows:
        img_df = pd.DataFrame(all_image_rows)
        img_df.to_csv(IMAGE_MANIFEST, index=False)
        label_counts = img_df["folder_label"].value_counts()
        print(f"\nImages: wrote {len(img_df)} image path(s) to "
              f"{os.path.relpath(IMAGE_MANIFEST, BASE_DIR)}.")
        print("        Candidate labels found (folder names) and counts:")
        for lbl, cnt in label_counts.items():
            print(f"          {lbl}: {cnt}")
        print("        These folder names are raw guesses from directory structure, not")
        print("        validated against the chatbot's fault categories — review before use.")
    else:
        print("\nImages: no image subfolders found in any dataset.")


if __name__ == "__main__":
    main()
