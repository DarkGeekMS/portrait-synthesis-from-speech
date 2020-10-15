# Portrait Synthesis From Speech

This repository contains the main code for `portrait synthesis from speech` project and its experiments. The project mainly consists of :
-   Speech Recognition.
-   Face generation from Text.
-   Face Refinement.
-   Multiple Head Poses Generation.

<div align="center" style="color:red;">
Refer to the wiki for more information about the project application, architecture and workflow.
</div>

## Installation

-   Install `python3` and `python3-pip`.

-   Install `requirements.txt` using `PyPi` :
    ```bash
    pip3 install -r requirements.txt
    ```

-   Alternatively, `Dockerfile` is provided with all required dependencies :
    ```bash
    docker build . -t portrait_syn_from_speech
    ```

-   Refer to individual `README.md` inside each component folder for further installation notes.
