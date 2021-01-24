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

## Generated Directions

-   `Blonde Hair Color Direction` : separates between __blonde hair__ _(+ve)_ and __dark hair__ _(-ve)_. It can generate a whole spectrum of hair colors. It can introduce some entanglement with skin color feature.

-   `Brown Hair Color Direction` : separates between __brown hair__ _(+ve)_ and __dark hair__ _(-ve)_. It can generate a whole spectrum of hair colors. It can introduce some entanglement with skin color feature.

-   `Gray Hair Color Direction` : separates between __white hair__ _(+ve)_ and __dark hair__ _(-ve)_. It can introduce heavy entanglement with age feature and some entanglement with skin color feature.

-   `Skin Color Direction` : separates between __dark skin__ _(+ve)_ and __light skin__ _(-ve)_. It does not completely preserve the identity and can introduce some entanglement with other features especially hair.

-   `Face Thickness Direction` : separates between __thick faces__ _(+ve)_ and __skinny faces__ _(-ve)_. It does not completely preserve the identity sometimes.

-   `Asian Effect Direction` : separates between __asian eyes__ _(+ve)_ and __normal eyes__ _(-ve)_. It does not completely preserve the identity and can introduce some entanglement with other features especially hair.

-   `Sight Glasses Direction` : separates between face __with__ _(+ve)_ and __without__ _(-ve)_ sight glasses. It does not completely preserve the identity and can introduce some entanglement with gender feature.

-   `Sun Glasses Direction` : separates between face __with__ _(+ve)_ and __without__ _(-ve)_ sun glasses. It does not completely preserve the identity.

-   `Baldness Direction` : separates between __bald people__ _(+ve)_ and __people with hair__ _(-ve)_. It does not completely preserve the identity sometimes and can introduce some entanglement with age feature.
