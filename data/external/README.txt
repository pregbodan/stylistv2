Place each downloaded Kaggle dataset's EXTRACTED FOLDER here, using these
exact directory names (an empty folder is already created for each):

  computer_hardware/        <- General Computer Hardware Dataset (Dilshaan Sandhu)
  electronic_components/    <- Electronic components and devices (aryaminus)
  electronic_parts/         <- Electronic Parts Dataset (olavomendes)
  software_reliability/     <- Software Reliability Dataset (vasanthkumarch)
  software_defects/         <- Software Defect Prediction (semustafacevik)

Copy the dataset's contents (CSVs, image subfolders, anything else inside
the Kaggle download) directly into the matching folder above — nesting can
be however Kaggle ships it; the ingestion script searches recursively.

Then from the project root, run:
  python3 ingest_external_datasets.py

See the "Incorporating the external Kaggle datasets" section of the main
README.md for what the script does and why it doesn't auto-merge results.
