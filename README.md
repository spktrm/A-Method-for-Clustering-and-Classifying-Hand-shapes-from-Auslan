# A-Method-Clustering-and-Classifying-Hand-shapes-from-Auslan

## How to use

### Step 1:
Use the get_signs.py script to create your data set or Auslan signs.

### Step 2:
Then use the create_data.py script to run the MediaPipe framework over your collected data to create a dataset for training a fully connected network.

### Step 3:
Next, use the make_hand_model.py script to train a full connected network. Be careful to examine the directory references.

### Step 4:
Finally, Use posenet_video.py script to classify your handshapes from data has been collected. Make sure this script references your trained hand model

## Optional 
For accuracy and performance metrics, use the classification_report.py script to see your models accuracy 

Scripts that end in "clustering.py" and "crop_hand.py" are used for clustering from a video source, where the number of hand shapes are unknown.
