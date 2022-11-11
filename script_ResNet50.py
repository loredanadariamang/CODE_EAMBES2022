import argparse
import borradorForTransferLearning_ResNet50 as BTF

# input argmuments
parser = argparse.ArgumentParser(description='Rocha: Lung Sound Classification')
parser.add_argument('--kfold_no', default=1, type=int, help='k fold number')
parser.add_argument('--feature', default='mel',help='type of time-freq representation')
parser.add_argument('--disease', default='128_96', help='window and overlap sizes')
parser.add_argument('--results_dir', default='RESULTS_mel/', help='directory fro storing the results')
parser.add_argument('--Kfold_dir', default='KFold_1/', help='K fold directory (from 1 to 5)')
args = parser.parse_args()

kfold_no = args.kfold_no
feature = args.feature                  # 'cocleogram' or 'stft' or 'mel'
disease = args.disease                  # 'crackles'  mean ALL TOGETHER -> CHANGE THIS LATER
results_dir = args.results_dir          # RESULTS_1/RESULTS_2/RESULTS_3/RESULTS_4/RESULTS_5/
kfold_dir = args.Kfold_dir              # KFol_1/KFol_2/KFol_3/KFol_4/KFol_5/

BTF.borrador(kfold_no, feature, disease, results_dir, kfold_dir)