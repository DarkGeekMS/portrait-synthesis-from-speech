# Dataset Preparation Utilities

This folder contains some utilities for dataset preparation using __face matching__ and __pseudo-text__.

## Datasets

-   [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset).
-   [Face2Text Dataset](https://drive.google.com/file/d/1cwcYbl0dhXEzmdbee_K_H6jcndbsxT2o/view).
-   [Face2Text with paraphrases Dataset](https://drive.google.com/file/d/12p8HR4HKyH16s0CR6-pC3_RrWxtlWeru/view?usp=sharing).
-   [Latent2Embedding Mapping Dataset](https://drive.google.com/file/d/1dQgFsYw3Faj6C3tsH8AnSrKhf9lLpSpN/view?usp=sharing).
-   [CelebA Dataset attributes csv with textual descriptions](https://drive.google.com/file/d/1Pw7myk-tj5CDEakHeRqvrL3Pj6Ap335Z/view?usp=sharing).

## Usage

-   Get top matches of a given set of faces from a database of faces :
    ```bash
    python get_face_matches.py -tj /path/to/text/json -fdi /path/to/target/faces/dir -fdb /path/to/face/database/dir -k similar_count
    ```

-   Filter top matches manually to get the best match :
    ```bash
    python filter_top_matches.py -jf /path/to/matches/json
    ```

-   Process dataset using _FaceNet_ :
    ```bash
    python extract_face_embed.py -facedir /path/to/dataset
    ```

-   Generate paraphrases for descriptions in _Face2Text_ dataset :
    ```bash
    python generate_paraphrases.py -tj /path/to/Face2TextDataset -pn max_number_of_paraphrases_to_be_generated
    ```

-   Generate textual descriptions for CelebA dataset using its attributes:
    ```bash
    python generate_CelebA_textual_descriptions.py -celebcsv /path/to/attribute/csv/file -p paraphrase_descriptions_to_re-structure_them_or_not_(0, 1)
    ```
