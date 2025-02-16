# Overview
The local_directory folder contains all Jupyter Notebooks that I used to plot the predictions and other visualizations. The _comp_cluster_directory_ is the directory I setup on the computational clusters of the Radboud University, the original path name is _/home/frewei/_ instead of _/comp_clusters_directory/_. Files that had to be plotted or visualized were moved to the _local_directory_ using the FileZilla program. The _/empty comp_clusters_directory/logs/_ folder correlates to _/local_directory/notebook_logs folder_.


_/comp_clusters_directory/single_source_loc/models/_ and _/comp_clusters_directory/multi_source_loc/models/_ are empty folders, as they exceeded the maximum upload limit of github. 

The final models were therefore placed in the following Google Drive: 
https://drive.google.com/drive/folders/1v9AhpLepNbx_jorwhvwbAESHQS7mYqEU?usp=drive_link

Older models that were trained with 100000 entries (SS4) and 140000 entries (SS5) are stored there as well, under _/pilot_test_files/_, (the weights of the SS4 and SS5 models could not be added, as the models take up too much memory.

# Virtual environment setup
The following packages have to be generated on the virtual environment:
````pip install tensorflow
pip install keras
pip install tensorboard
pip install skicit
pip install regex
pip install numpy
pip install matplotlib
````
The virtual environment was named fre_env.

# Pipeline
(The following steps are meant for the Multi Source Models, but can be similarly implemented for the Single Source Models. The scripts are meant to run on the computational clusters Earth/Mars. Check what GPU will be used in the script before running the commands, and edit the number in the scripts.

1. Create snap-shot of current files in folders
````
source /home/frewei/fre_env/bin/activate
python /home/frewei/single_source_loc/MultiNamesLabelsDef.py
deactivate
````
2. Train model
````
source /home/frewei/fre_env/bin/activate
nohup python /home/frewei/multi_source_loc/NewMultiSource1_same_BS_8_LR_1e-5.py > /home/frewei/multi_source_loc/script_outputs/NewMS1_same_BS_8_LR_1e-5.out 2>&1 &
deactivate
````
3. Predict model
````
source /home/frewei/fre_env/bin/activate
python /home/frewei/single_source_loc/PredictMultiModel.py
deactivate
````




