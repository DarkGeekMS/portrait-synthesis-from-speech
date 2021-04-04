# Portrait Synthesis From Speech

<div align=center><img width="50%" height="50%" src="logo.png"/></div>

--------------------------------------------------------------------------------

This repository contains the main code for `portrait synthesis from speech` experiments associated with [Retratista](https://github.com/DarkGeekMS/Retratista) application.

## Dependencies

-   Install `python3` and `python3-pip`.

-   Install `requirements.txt` using `PyPi` :
    ```bash
    pip3 install -r requirements.txt
    ```

-   Alternatively, `Dockerfile` is provided with all required dependencies :
    ```bash
    # build image from dockerfile
    docker build . -t portrait_syn_from_speech:latest
    # run container from image in interactive session
    docker run --runtime=nvidia -it portrait_syn_from_speech:latest /bin/bash
    ```

-   Refer to individual `README.md` inside each component folder for further installation notes.

## System Architecture

<div align=center><img width="100%" height="100%" src="architecture.png"/></div>

<div align="center">
Figure(1): Complete system architecture diagram showing the flow between different modules.
</div><br>

__Refer to the wiki for more information about the project application, architecture and workflow.__
