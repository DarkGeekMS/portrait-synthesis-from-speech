# Attributes Axes (Directions) Generation

This folder contains the code for different attributes axes (directions) generation.

## Usage

-   Run manual filter of the generated images (2 classes only) :
    ```bash
    python manual_filter.py -id /path/to/images/dir -c1d /path/to/class1/output/dir \
    -c2d /path/to/class2/output/dir
    ```
__NOTE :__ Use `a` key to add image to class 1 and `d` key to add image to class 2.

-   Run axis fit on prepared classes :
    ```bash
    python fit_axis.py -c1d /path/to/class1/output/dir \
    -c2d /path/to/class2/output/dir -dn <output_file_name>
    ```

-   Run axes disentanglement on 2 specific axes :
    ```bash
    python disentangle_axes.py -sa /path/to/source/axis/file \
    -ta /path/to/target/axis/file
    ```

-   Run manual data labelling on images with seed _[5700:6700]_ :
    ```bash
    python save_data.py
    ```

## Generated Directions

Refer to [wiki](https://github.com/DarkGeekMS/portrait-synthesis-from-speech/wiki/Complete-Set-of-Considered-Facial-Attributes) for complete set of considered facial features. Also, refer to [wiki](https://github.com/DarkGeekMS/portrait-synthesis-from-speech/wiki/Generated-Directions-Results) for visual samples of generated feature directions results.