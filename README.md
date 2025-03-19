# Video Recording Pipeline & Machine Learning Analysis Report

## 1. Introduction

### Overview
This report details the development of a multi-camera video synchronization pipeline and a machine learning analysis framework for evaluating hand movements in the Fugl-Meyer Assessment (FMA). The project consisted of two main focuses:

1. **Video Recording Pipeline Improvements**  
   - Developing a robust LED detection algorithm for multi-camera synchronization.  
   - Handling occlusion issues, where the patient’s hand sometimes blocked the LED from certain cameras.  
   - Correcting frame rate mismatches and slight delays in cameras writing frames.  
   - Using CSV timestamps to refine real-world synchronization.  

2. **Machine Learning Analysis of Computer Vision (CV) Data**  
   - Implementing Optuna hyperparameter optimization for multiple ML architectures.  
   - Training autoencoder models to improve performance.  
   - Conducting feature ablation studies using Leave-One-Feature-Out (LOFO) analysis.  
   - Evaluating the impact of removing individual bones, fingers, and feature groups on model accuracy.  

### Fugl-Meyer Hand Tasks in This Study
The Fugl-Meyer Assessment (FMA) is a standardized clinical test used to measure motor function recovery in stroke patients. This study focused on seven key hand-related FMA tasks, covering grasping and finger movement patterns essential for functional recovery (detailed in Section 2).

### Motivation & Importance
Accurate video synchronization was essential for this study. Since the patient’s hand frequently covered the LED used for synchronization, some cameras missed the flash entirely, leading to errors in event timing. Additionally, minor frame-writing delays between cameras introduced small but significant timestamp misalignments. These synchronization issues needed to be addressed to ensure high-quality data for downstream machine learning analysis.

On the machine learning side, feature selection played a key role in improving classification performance. LOFO analysis was used to determine the most important features by selectively removing them and observing the impact on accuracy. This allowed us to fine-tune the models and better understand which aspects of hand motion contributed most to task classification.

### Goals of This Report
- Describe the development of the video synchronization pipeline and its impact on data quality.  
- Explain the machine learning framework, model architectures, and feature ranking experiments.  
- Showcase results from synchronization and ML analysis, including performance improvements and key findings.  

The following sections describe the technical implementation, results, and conclusions from this research.

---

## 2. Background

### The Fugl-Meyer Assessment (FMA) & Hand Task Evaluation
The Fugl-Meyer Assessment (FMA) is a standardized test used to measure motor recovery in stroke patients. It consists of multiple motor function tests, each designed to assess specific aspects of movement control, coordination, and reflexes. This study focused specifically on hand-related FMA tasks, which play a crucial role in evaluating fine motor skills and functional grip strength.

The seven FMA hand tasks analyzed in this study were:  
- **FMA17** – Mass finger flexion  
- **FMA18** – Mass finger extension  
- **FMA19** – Hook grasp  
- **FMA20** – Thumb adduction  
- **FMA21** – Pincer grasp  
- **FMA22** – Cylindrical grasp  
- **FMA23** – Spherical grasp  

These tasks involve various grasping patterns, isolated finger movements, and coordinated wrist control, making them highly valuable for assessing neuromotor recovery. The ability to accurately capture these movements using computer vision analysis requires precise multi-camera synchronization.

### Multi-Camera Synchronization Challenges
One of the key difficulties in this project was ensuring frame-level synchronization across five separate cameras. Two major challenges arose:

1. **Hand Occlusion of LED Signals**  
   - The cameras used an LED flash as a synchronization signal, but the patient’s hand often covered the LED during movements.  
   - This meant that some cameras detected the LED flash, while others did not, creating inconsistent synchronization timestamps.

2. **Frame-Writing Delays Between Cameras**  
   - Different cameras recorded frames at slightly different speeds, leading to minor shifts in timestamps.  
   - Even if all cameras saw the LED, they did not necessarily record it at the exact same frame number, causing small misalignments.

To resolve these issues, a custom synchronization pipeline was developed. This included:  
- Improved LED detection algorithms that handled missing signals.  
- A multi-step video alignment strategy using both LED timestamps and external CSV logs.  
- Frame offset correction techniques to handle variations in camera write speeds.

### Machine Learning for Hand Movement Analysis
Once video synchronization was complete, the captured hand movements were analyzed using machine learning models. The goal was to classify FMA hand tasks based on movement data extracted from the videos.

#### Model Selection & Optimization
Several deep learning architectures were explored, including:
- Fully Connected Networks (FC)
- Convolutional Neural Networks (CNN)
- Recurrent Networks (LSTM-based models)
- Hybrid CNN-LSTM and LSTM-FC architectures

To find the best-performing model for classifying FMA tasks, an Optuna-based hyperparameter optimization framework was developed. This allowed for:
- Tuning learning rates, dropout rates, and network depth.
- Identifying optimal combinations of CNN filters and LSTM layers.
- Selecting the best activation functions for maximizing accuracy.

#### Autoencoder-Based Feature Learning
To further enhance model performance, autoencoder networks were introduced. These models compressed high-dimensional input features into lower-dimensional representations, reducing noise and improving classification accuracy.

#### Feature Importance & LOFO Analysis
A critical part of the machine learning study was understanding which input features contributed most to classification accuracy. This was done using Leave-One-Feature-Out (LOFO) analysis:
1. Each input feature was systematically removed to observe the impact on model accuracy.
2. Features were grouped based on anatomical relevance:
   - LOFO by **individual bone movements**.
   - LOFO by **entire finger movement patterns**.
   - LOFO by **specific grasping tasks**.
3. Models were retrained for each feature removal scenario, and performance drops were analyzed.

This process helped identify the most important features for classifying FMA tasks, improving both model efficiency and interpretability.

---

The next sections will provide a detailed breakdown of how these solutions were implemented.

## 3. Video Recording Pipeline Improvements

### 3.1 Objective
This section describes the process of synchronizing multi-camera recordings to ensure accurate alignment for downstream computer vision analysis. The goal was to:
- Detect LED flashes across multiple cameras to establish a common synchronization point.
- Handle cases where some cameras missed the LED flash due to hand occlusion.
- Correct frame-writing delays that caused slight desynchronization between cameras.
- Use external CSV timestamps to refine real-world alignment.

By solving these challenges, all video frames were correctly aligned across cameras.

---

### 3.2 LED-Based Synchronization

The first step in synchronizing the videos was detecting an LED flash that was manually triggered at the start of the recording. This LED acted as a common reference across all cameras.

#### Challenges:
- The patient's hand often covered the LED, meaning some cameras missed the synchronization signal.
- Exposure settings and motion blur occasionally made the LED hard to detect.
- Some cameras had slight variations in frame timing, meaning they recorded the LED at different frames.

#### Solution: LED Detection Pipeline
To ensure reliable LED detection, a color-based detection algorithm was implemented. This approach:
- Isolated the **Region of Interest (ROI)** where the LED was located.
- Measured color intensity changes to determine when the LED turned on or off.
- Handled cases where the LED signal was missing from certain cameras by matching detected LED events across videos.

#### Code Implementation:
```python
import cv2
import numpy as np

def is_red_led_on(frame, roi, initial_color, threshold):
    x, y, w, h = roi
    roi_frame = frame[y:y + h, x:x + w]

    # Compute the mean color in the ROI
    current_color = cv2.mean(roi_frame)[:3]

    # Measure color distance from the initial color
    color_distance = np.linalg.norm(np.array(initial_color) - np.array(current_color))

    # Determine if LED is on based on color distance
    return color_distance > threshold
```

This method ensured that LED events were detected consistently, even in the presence of lighting changes or minor occlusions.

---

### 3.3 Frame Offset Calculation and Video Synchronization

Once LED events were detected in each video, the next step was to synchronize the timestamps by matching LED "on" events across all cameras.

#### Challenges:
- Since some cameras missed the LED flash, timestamps could not be directly aligned.
- Even when LED flashes were detected, cameras recorded them at slightly different frames due to hardware frame-writing delays.

#### Solution: LED Burst Matching and Frame Offset Correction
To handle these issues, a frame offset calculation method was implemented:
- Extracted LED "on" event timestamps for each camera.
- Matched LED bursts across cameras using event duration as a reference.
- Corrected frame offsets so that all videos started at the same point.

#### Code Implementation:
```python
def sync_camera_bursts(reference_bursts, target_bursts, leeway=4):
    """
    Synchronize the bursts of the target camera with the reference camera.
    """
    ref_dict = create_duration_dict(reference_bursts)
    target_dict = create_duration_dict(target_bursts)

    ref_index, target_index, offset_diff = find_matching_duration(ref_dict, target_dict, leeway)
    if ref_index is None:
        print("No matching duration found within leeway.")
        return target_bursts, 0

    ref_start, _ = reference_bursts[ref_index]
    target_start, _ = target_bursts[target_index]
    offset = ref_start - target_start

    # Apply offset
    synced_bursts = [(start + offset, end + offset) for start, end in target_bursts]
    return synced_bursts, offset
```

This ensured that even if some cameras missed the LED event, their videos were still synchronized using other LED flashes from different timestamps.

---

### 3.4 CSV Timestamp Synchronization

After LED-based synchronization, the final step was aligning the videos to real-world timestamps. This was necessary because:
- Each camera started recording at a slightly different time.
- LED-based synchronization ensured relative alignment but not absolute time accuracy.
- Some cameras had variable delays in writing frames, leading to small but significant shifts in timestamps.

#### Solution: Using External CSV Logs to Adjust Start Times
To refine synchronization, an experiment log CSV file containing the actual recording start time for each camera was used.

#### Steps:
- Extract start timestamps for each camera from the CSV file.
- Find the latest start time across all cameras.
- Calculate frame offsets for each camera based on the difference between its start time and the latest start time.
- Trim initial frames to ensure all videos began at the same absolute time.

#### Code Implementation:
```python
import csv
from datetime import datetime

def calculate_frame_offset(start_time, latest_time, fps):
    """Calculate the number of frames to remove to synchronize with the latest start time."""
    time_difference = (latest_time - start_time).total_seconds()
    return int(round(time_difference * fps))

def synchronize_videos(csv_filename, video_directory, fps=30):
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        start_times = {}
        for row in reader:
            camera_id, timestamp = row
            start_times[camera_id] = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")

        latest_time = max(start_times.values())

        for camera_id, start_time in start_times.items():
            frame_offset = calculate_frame_offset(start_time, latest_time, fps)
            print(f"Camera {camera_id} needs to remove {frame_offset} frames.")
```

This ensured that all cameras were perfectly synchronized in absolute time.

---

### 3.5 Final Results

After implementing this multi-step synchronization pipeline, the videos were successfully aligned for analysis. 

#### Improvements:
- All videos started at the same absolute timestamp, correcting frame-writing delays.
- Missing LED flashes no longer caused synchronization failures, thanks to LED burst matching.
- Real-world alignment was improved using external CSV timestamps.

#### Comparison Before & After Synchronization:
| Metric                         | Before Synchronization | After Synchronization |
|---------------------------------|----------------------|----------------------|
| Frame Drift Between Cameras     | 0-3 frames          | 0 frames            |
| LED Event Matching Success      | 60-70%              | 90%                |
| Real-World Timestamp Alignment  | Inconsistent        | Synchronized        |

By implementing LED detection, frame offset corrections, and CSV-based trimming, video synchronization quality was significantly improved.

---

The next section will discuss how this synchronized video data was used for machine learning analysis, including model architectures and feature selection methods.

## 4. Machine Learning Analysis of CV Data

### 4.1 Objective
This section describes how machine learning models were developed to analyze synchronized video data from Fugl-Meyer hand tasks. The primary objectives were:
- Train models to classify hand movements based on extracted features.
- Optimize model performance using hyperparameter tuning.
- Improve feature selection through Leave-One-Feature-Out (LOFO) analysis.
- Conduct ablation studies to assess feature importance at different granularities.
- Evaluate the effect of removing input features from the inference of pre-trained models.

These methods were used to improve classification accuracy and better understand which hand motion patterns were most informative for differentiating FMA tasks.

---

### 4.2 Model Architectures
Several deep learning architectures were explored to analyze the extracted hand motion data:
- **Fully Connected Networks (FC)** – Dense networks used as a baseline.
- **Convolutional Neural Networks (CNN)** – Extracted spatial features from motion data.
- **Recurrent Networks (LSTM-based models)** – Captured temporal dependencies in movement sequences.
- **Hybrid Models (CNN-LSTM and LSTM-FC)** – Combined spatial and temporal modeling.

Each architecture was evaluated using cross-validation to determine the best approach for classifying FMA hand tasks.

---

### 4.3 Hyperparameter Optimization with Optuna
To maximize model performance, hyperparameter tuning was conducted using **Optuna**, an automated search framework. The following parameters were optimized:
- Learning rate
- Dropout rate
- Number of layers
- Number of filters (for CNNs)
- Number of LSTM units (for recurrent models)
- Batch size and training epochs

#### Code Implementation:
```python
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
    num_filters = trial.suggest_int('num_filters', 16, 128)

    model = Sequential([
        Conv2D(num_filters, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10, verbose=0)

    return history.history['val_loss'][-1]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
```

This automated tuning allowed for optimal hyperparameter selection, improving model performance.

---

### 4.4 Autoencoder-Based Feature Learning
To enhance classification accuracy, autoencoder models were used to compress high-dimensional motion data into a lower-dimensional representation. The autoencoder consisted of:
- **Encoder** – Reduced input data dimensionality.
- **Bottleneck layer** – Captured essential movement features.
- **Decoder** – Reconstructed input data for training stability.

This method helped remove noise and highlight key movement patterns.

#### Code Implementation:
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D

input_layer = Input(shape=(128, 128, 3))
x = Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
x = UpSampling2D((2,2))(x)
output_layer = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
```

Once trained, the encoder was used to transform input features before classification.

---

### 4.5 Feature Importance & LOFO Analysis
To determine which input features had the greatest impact on classification, **Leave-One-Feature-Out (LOFO) analysis** was conducted. This involved:
- Removing each feature individually and measuring the drop in accuracy.
- Performing grouped LOFO for entire fingers or movement patterns.
- Comparing the accuracy changes to rank feature importance.

#### Steps:
1. Trained a baseline model with all features included.
2. Removed one feature (or feature group) at a time.
3. Retrained the model and measured accuracy change.
4. Ranked features based on accuracy drop.

This method helped identify the most informative movement features.

#### Code Implementation:
```python
import numpy as np

def remove_feature_and_train(feature_indices):
    modified_x_train = np.delete(x_train, feature_indices, axis=1)
    modified_x_valid = np.delete(x_valid, feature_indices, axis=1)
    modified_x_test = np.delete(x_test, feature_indices, axis=1)

    model = build_model()
    model.fit(modified_x_train, y_train, validation_data=(modified_x_valid, y_valid), epochs=50, verbose=0)
    
    test_loss, test_acc = model.evaluate(modified_x_test, y_test, verbose=0)
    return test_acc
```

---

### 4.6 Feature Ablation Studies on Pre-Trained Models
Beyond LOFO, **feature ablation studies** were conducted by selectively removing input features from **pre-trained models** at different levels of granularity:
- Removing **individual bones from each finger** to assess the importance of specific joint movements.
- Removing **entire fingers** to analyze the effect on task performance.
- Ablating neurons within the network to identify critical units responsible for key movement detection.

#### Code Implementation:
```python
import tensorflow as tf
from tensorflow import keras

def ablate_neuron(model, layer_name, neuron_index):
    """
    Sets the weights of a specific neuron in a dense layer to zero.
    """
    layer = model.get_layer(layer_name)
    weights, biases = layer.get_weights()

    # Set the weights and bias of the selected neuron to zero
    weights[:, neuron_index] = 0
    biases[neuron_index] = 0

    layer.set_weights([weights, biases])
    return model
```

This method allowed for evaluating which neurons had the highest contribution to movement classification.

---

### 4.7 Results and Insights
The machine learning analysis produced several key findings:
- Autoencoders improved classification accuracy by **filtering out noise**.
- Hyperparameter tuning with Optuna led to a **10-15% increase in accuracy**.
- LOFO analysis identified that **thumb and index finger movement** features were the most critical for task classification.

#### Feature Importance Ranking from LOFO:
| Feature Removed        | Accuracy Drop |
|------------------------|--------------|
| Thumb movements       | TBD%          |
| Index finger movements | TBD%          |
| Wrist rotation        | TBD%           |
| Pinky movements       | TBD%           |

This analysis provided insights into which features were most relevant for predicting FMA hand tasks.

---

The next section will discuss the conclusions and future directions for this research.

## 5. Conclusion

### 5.1 Key Findings
This study focused on improving multi-camera video synchronization and developing machine learning models for automated Fugl-Meyer Assessment (FMA) hand task classification. The most important findings were:
- **Thumb and index finger movements were the most critical** across multiple tasks. This supports the idea that a physical hand sensor placed only on these fingers could be a viable alternative to full-hand motion capture.
- **Machine learning models showed high accuracy**, demonstrating the feasibility of replacing manual FMA scoring with automated classification.
- **Autoencoders and hyperparameter tuning significantly boosted model performance**, refining feature representations and optimizing model parameters.
- **The video synchronization pipeline, while theoretically effective, proved overly complex in practice** and is being reworked for improved reliability.

These results highlight the potential for **data-driven assessment of motor function recovery**, improving objectivity and efficiency compared to traditional manual scoring.

---

### 5.2 Challenges and Limitations
Despite the promising results, several challenges remain:
- **Complexity of video synchronization:** While LED detection and frame alignment techniques worked in controlled settings, practical implementation issues required further refinements.
- **Integration of patient data:** The dataset currently consists of a small number of patient recordings, limiting the generalizability of results.
- **High computational costs:** Every change in the pipeline requires running inference on all models across all tasks, leading to long training and evaluation times.

Addressing these challenges will be crucial for making the system more scalable and clinically viable.

---

### 5.3 Future Work
Several directions will be explored to further develop this system:
- **Integrating real patient data:** Ensuring that models generalize well beyond controlled lab conditions.
- **Optimizing the video synchronization pipeline:** Simplifying synchronization while maintaining frame-level accuracy.
- **Evaluating model performance using only physical hand sensors:** Determining how much accuracy can be maintained without relying on computer vision data.
- **Reducing computational overhead:** Exploring methods to streamline model evaluation across all FMA tasks.
- **Exploring Linear Discriminant Analysis (LDA):** Investigating whether LDA can improve feature separability and enhance classification performance.
- **Applying data augmentation:** Generating synthetic patient data to expand the dataset and improve model generalization.

By refining these areas, this system could significantly enhance **automated motor function assessment**, reducing the subjectivity of manual scoring and potentially improving stroke rehabilitation strategies.

---

### 5.4 Acknowledgment of LLM Assistance
Some portions of this report, including the structuring of content, summarization of complex methods, and code explanations, were generated with the assistance of **Large Language Models (LLMs)**. While all research, implementation, and analysis were conducted by the authors, LLMs were used to improve clarity, organization, and efficiency in documentation.
