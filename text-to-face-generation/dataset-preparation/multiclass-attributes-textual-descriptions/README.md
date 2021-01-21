# Dataset Preparation Utilities

This folder contains utilities for dataset generation with attributes and textual descriptions in a __multi-class__, __multi-label__ form.

## Usage

-   Generate a csv file contains random attributes without a description :
    ```bash
    python generate_attribute_csv.py -n_records num_of_records
    ```

-   Then generate the descriptions:
    ```bash
    python generate_attributes_descriptions.py -attrcsv path/to/attributes/csv/file -p paraphrase_choice
    ```