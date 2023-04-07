## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

## This code is modified by Linjie Li from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa
## GNU General Public License v3.0

## Script for downloading data

# VQA Questions
import wget
import zipfile
import os

# wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
# zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Questions_Train_mscoco.zip')
# zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\data\Questions")
# os.remove("E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Questions_Train_mscoco.zip")

# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
# unzip data/v2_Questions_Val_mscoco.zip -d data/Questions
# rm data/v2_Questions_Val_mscoco.zip
# wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
# zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Questions_Val_mscoco.zip')
# zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\data\Questions")
# zip_file.close()
# os.remove("E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Questions_Val_mscoco.zip")
# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
# unzip data/v2_Questions_Test_mscoco.zip -d data/Questions
# rm data/v2_Questions_Test_mscoco.zip
# wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
# zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Questions_Test_mscoco.zip')
# zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\data\Questions")
# zip_file.close()
# os.remove("E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Questions_Test_mscoco.zip")
# # VQA Annotations
# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
# unzip data/v2_Annotations_Train_mscoco.zip -d data/Answers
# rm data/v2_Annotations_Train_mscoco.zip
# unzip data/v2_Annotations_Val_mscoco.zip -d data/Answers
# rm data/v2_Annotations_Val_mscoco.zip
#

wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Annotations_Train_mscoco.zip')
zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\data\Answers")
zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Annotations_Val_mscoco.zip')
zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\data\Answers")
zip_file.close()
os.remove("E:\PycharmProjects\VQA_ReGAT-master\data\\v2_Annotations_Val_mscoco.zip")





# # VQA cp-v2 Questions
# mkdir data/cp_v2_questions
# wget -P data/cp_v2_questions https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json
# wget -P data/cp_v2_questions https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json


wget.download("https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json", 'E:\PycharmProjects\VQA_ReGAT-master\data\cp_v2_questions')
wget.download("https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json", 'E:\PycharmProjects\VQA_ReGAT-master\data\cp_v2_questions')



# # VQA cp-v2 Annotations
# mkdir data/cp_v2_annotations
# wget -P data/cp_v2_annotations https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json
# wget -P data/cp_v2_annotations https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json
wget.download("https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json", 'E:\PycharmProjects\VQA_ReGAT-master\data\cp_v2_annotations')
wget.download("https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json", 'E:\PycharmProjects\VQA_ReGAT-master\data\cp_v2_annotations')




# # Visual Genome Annotations
# mkdir data/visualGenome
# wget -P data/visualGenome https://convaisharables.blob.core.windows.net/vqa-regat/data/visualGenome/image_data.json
# wget -P data/visualGenome https://convaisharables.blob.core.windows.net/vqa-regat/data/visualGenome/question_answers.json
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/visualGenome/image_data.json", 'E:\PycharmProjects\VQA_ReGAT-master\data\\visualGenome')
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/visualGenome/question_answers.json", 'E:\PycharmProjects\VQA_ReGAT-master\data\\visualGenome')



# # GloVe Vectors and dictionary
# wget -P data https://convaisharables.blob.core.windows.net/vqa-regat/data/glove.zip
# unzip data/glove.zip -d data/glove
# rm data/glove.zip

wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/glove.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\vqa-regat/data/glove.zip')
zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\data\glove")
zip_file.close()
os.remove("E:\PycharmProjects\VQA_ReGAT-master\data\\vqa-regat/data/glove.zip")


# # Image Features
# # adaptive
# # WARNING: This may take a while
# mkdir data/Bottom-up-features-adaptive
# wget -P data/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/train.hdf5
# wget -P data/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/val.hdf5
# wget -P data/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/test2015.hdf5

wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/train.hdf5", 'E:\PycharmProjects\VQA_ReGAT-master\data\Bottom-up-features-adaptive')
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/val.hdf5", 'E:\PycharmProjects\VQA_ReGAT-master\data\Bottom-up-features-adaptive')
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/test2015.hdf5", 'E:\PycharmProjects\VQA_ReGAT-master\data\Bottom-up-features-adaptive')



# # fixed
# # WARNING: This may take a while
# mkdir data/Bottom-up-features-fixed
# wget -P data/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/train36.hdf5
# wget -P data/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/val36.hdf5
# wget -P data/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/test2015_36.hdf5

wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/train36.hdf5", 'E:\PycharmProjects\VQA_ReGAT-master\data\Bottom-up-features-fixed')
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/val36.hdf5", 'E:\PycharmProjects\VQA_ReGAT-master\data\Bottom-up-features-fixed')
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/test2015_36.hdf5", 'E:\PycharmProjects\VQA_ReGAT-master\data\Bottom-up-features-fixed')


# # imgids
# wget -P data/ https://convaisharables.blob.core.windows.net/vqa-regat/data/imgids.zip
# unzip data/imgids.zip -d data/imgids
# rm data/imgids.zip
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/imgids.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\imgids.zip')
zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\data\imgids")
zip_file.close()
os.remove("E:\PycharmProjects\VQA_ReGAT-master\data\\imgids.zip")


# # Download Pickle caches for the pretrained model
# # and extract pkl files under data/cache/.
# wget -P data https://convaisharables.blob.core.windows.net/vqa-regat/data/cache.zip
# unzip data/cache.zip -d data/cache
# rm data/cache.zip
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/data/cache.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\cache.zip')
zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\data\cache")
zip_file.close()
os.remove("E:\PycharmProjects\VQA_ReGAT-master\data\\cache.zip")



# # Download pretrained models
# # and extract files under pretrained_models.
# wget https://convaisharables.blob.core.windows.net/vqa-regat/pretrained_models.zip
# unzip pretrained_models.zip -d pretrained_models/
# rm pretrained_models.zip
wget.download("https://convaisharables.blob.core.windows.net/vqa-regat/pretrained_models.zip", 'E:\PycharmProjects\VQA_ReGAT-master\data')
zip_file = zipfile.ZipFile('E:\PycharmProjects\VQA_ReGAT-master\data\\pretrained_models.zip')
zip_file.extractall("E:\PycharmProjects\VQA_ReGAT-master\pretrained_models")
zip_file.close()
os.remove("E:\PycharmProjects\VQA_ReGAT-master\data\\pretrained_models.zip")