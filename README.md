 # Face Mask Detection                   
                   
This project uses a Faster R-CNN model to detect whether faces are wearing masks. The idea came from a Kaggle discussion and the notebook by [alperkaraca1](https://www.kaggle.com/code/alperkaraca1/faster-r-cnn-face-mask-detection), released under the Apache 2.0 license. The dataset is from [Andrew Mvd’s Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/data).

## How It Works
1. **Dataset Parsing**: Loads images and XML annotations with bounding boxes labeled “with_mask” or “without_mask.”
2. **Model Training**: Uses a pretrained Faster R-CNN ResNet50 FPN from PyTorch, fine-tunes it for mask detection.
3. **Real-Time Detection**: Runs the trained model on a webcam feed to draw bounding boxes and label faces.

## Usage
1. Place images and annotations under `data/images` and `data/annotations`.
2. Run `python train_model.py` to train (creates `fasterrcnn_model.pth`).
3. Run `python real_time_detection.py` to start the webcam. Press Esc or q to exit.

note: still need some work for better detection accuracy
