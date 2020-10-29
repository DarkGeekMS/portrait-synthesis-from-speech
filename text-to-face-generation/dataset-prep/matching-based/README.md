# Dataset Preparation Utilities (Face Matching)

This folder contains some utilities for dataset preparation using __face matching__.

## Datasets

-   [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset).
-   [Face2Text Dataset](https://drive.google.com/file/d/1cwcYbl0dhXEzmdbee_K_H6jcndbsxT2o/view).

## Usage

-   Get top matches of a given set of faces from a database of faces :
    ```bash
    python get_face_matches.py -tj /path/to/text/json -fdi /path/to/target/faces/dir -fdb /path/to/face/database/dir -k similar_count
    ```

-   Visualize all face matches :
    ```bash
    python viz_match.py -jf /path/to/matches/json
    ```

-   Filter top matches manually to get the best match :
    ```bash
    python filter_top_matches.py -jf /path/to/matches/json
    ```

-   Process dataset using Facenet :
    ```bash
    python process_dataset.py -facedir /path/to/dataset
    ```

-   Generate paraphrases for descriptions in Face2Text dataset :
    ```bash
    python paraphrase_generation.py -tj /path/to/Face2TextDataset -pn max_number_of_paraphrases_to_be_generated
    ```
