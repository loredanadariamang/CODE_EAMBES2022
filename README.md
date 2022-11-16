# CODE_for_Cochleogram-based_Adventitious_Sounds_Classification_using_Convolutional_Neural_Networks

 In order to be able to run this code, several folders are needed:
 - Graphs: a folder that will store the figures for the CNN models in each experiment.
 - MODELS: a folder that will store the models and weights generated in each experiment.
 - RESULTS: a folder that will store .npz files with the results generated in each experiment. In our experiments we achieved results from different time-frequency representations, so this folder was subdivided into several folders like: "Cochleogram/Mel/STFT..".
 - KFOLD_IDX: a folder that will save the positions within each train/test set for each fold (in this case we experimented with a 10 K-fold ran 5 times - see "KFold.py" code).
 - processed_audio_files: a folder that will contain the breathing cicles obtained by running the preprocessing code when applied to the ICBHI_Database that can be downloaded from: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
 
 For any aditional information please contact with the repository owner.
 Please cite us if you reference our materials.
