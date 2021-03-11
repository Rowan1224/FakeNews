pip install gdown
mkdir Data
cd Data
gdown https://drive.google.com/uc?id=1ylDPvR0f3ggZDsQ6Z2KMTydIgfIoLksP

mkdir Word2Vec
cd Word2Vec
gdown https://drive.google.com/uc?id=1FedGrRGuHsZSCAmKwuD6yc4Bl-qIb-2R
gdown https://drive.google.com/uc?id=17uCa4eRmYrUnD3n5NCjLbl4AxUByqikm

cd ..
mkdir POS
cd POS
gdown https://drive.google.com/uc?id=1Rm5c1FJtmQnnRCfupYsFa20geHwX2_ud

cd ..
mkdir AlexaRank
cd AlexaRank
gdown https://drive.google.com/uc?id=1U-dlrfnnwILjUFLUbUiIZ8Xa95bmVFte
gdown https://drive.google.com/uc?id=1rekfv4DzdoxFGJMIAWIX1hMNjtuOzGDM
gdown https://drive.google.com/uc?id=1GxM2bj_Z0Yuq1zKSAhhhJexyyJ1gm_Rl
gdown https://drive.google.com/uc?id=1v4EwGvGywpJFcV6EREpt3Q30G4ILdBRJ

cd ..
cd data_processing
python train_test.py
