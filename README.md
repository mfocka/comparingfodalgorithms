# COMPARING FOD ALGORITHMS

This project is a Python-based object detection application.

## Folder Structure

- **BASLER:** Put pictures taken on the Basler Dart here.
- **OAK:** Put pictures taken on the OAK-1 here.
- **OTHER:** Put other pictures here.

- **detectors:** Each object detection model has its own class/file in this folder.

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/mfocka/comparingfodalgorithms.git
   ```

2. Set the algorithm you want to run in `app.py`:

    ```python
    model = MODELNAME(ml, ml, '.tflite', i/20, onEdge=False, doTpu=False)
    test_model(model)
    ```

3. Run `app.py`:

    ```sh
    python app.py
    ```
