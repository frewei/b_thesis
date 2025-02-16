# Overview
The local_directory folder contains all Jupyter Notebooks that I used to plot the predictions and other visualizations. The _comp_cluster_directory_ is the directory I setup on the computational clusters of the Radboud University, the original path name is _/home/frewei/_ instead of _/comp_clusters_directory/_. Files that had to be plotted or visualized were moved to the _local_directory_ using the FileZilla program. The _/empty comp_clusters_directory/logs/_ folder correlates to _/local_directory/notebook_logs folder_.


_/comp_clusters_directory/single_source_loc/models/_ and _/comp_clusters_directory/multi_source_loc/models/_ are empty folders, as they exceeded the maximum upload limit of github. 

The final models were therefore placed in the following Google Drive: 
https://drive.google.com/drive/folders/1v9AhpLepNbx_jorwhvwbAESHQS7mYqEU?usp=drive_link

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

# Layout folders
The folder university_directory is the home folder of my account 'frewei' on the computational clusters of Radboud University. Files that had to be plotted or visualized were moved to the local_directory using the FileZilla program. The /empty comp_clusters_directory/logs/ folder correlates to /local_directory/notebook_logs

Pilot test models and predictions are placed in their predictions and models folders withing the single_source_loc and multi_source_loc folders

predictions van pilot tests werd gedaan in de model python script, daarvoor is de predictions file in single_source_loc


````
source /home/frewei/fre_env/bin/activate
nohup python /home/frewei/multi_source_loc/NewMultiSource1_same_BS_8_LR_1e-5.py > /home/frewei/multi_source_loc/script_outputs/NewMS1_same_BS_8_LR_1e-5.out 2>&1 &
deactivate
````

````
source /home/frewei/fre_env/bin/activate
python /home/frewei/single_source_loc/SingleNamesLabelsDef.py
deactivate
````

````
source /home/frewei/fre_env/bin/activate
python /home/frewei/single_source_loc/PredictMultiModel.py
deactivate
````
