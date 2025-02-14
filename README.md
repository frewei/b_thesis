# b_thesis

necesarry packages:
````pip install tensorflow
pip install keras
pip install tensorboard
pip install skicit
pip install regex
pip install numpy
pip install matplotlib
````
# Layout folders
The folder university_directory is the home folder of my account 'frewei' on the computational clusters of Radboud University. Files that had to be plotted or visualized were moved to the local_directory using the FileZilla program. THe following (empty) folders correlate with the university_directory folders:
logs with logs, prediction

Pilot test models and predictions are placed in their predictions and models folders withing the single_source_loc and multi_source_loc folders

predictions van pilot tests werd gedaan in de model python script, daarvoor is de predictions file in single_source_loc


````
source /home/frewei/fre_env/bin/activate
nohup python /home/frewei/multi_source_loc/NewMultiSource1_same_BS_8_LR_1e-5.py > /home/frewei/multi_source_loc/script_outputs/NewMS1_same_BS_8_LR_1e-5.out 2>&1 &
deactivate
````
