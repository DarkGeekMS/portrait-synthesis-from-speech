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

-   Filter top matches manually to get the best match :
    ```bash
    python filter_top_matches.py -jf /path/to/matches/json
    ```
