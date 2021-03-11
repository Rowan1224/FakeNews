
EMBEDDING_300 = '../../Data/Word2Vec/cc.bn.300.vec'
EMBEDDING_100 = '../../Data/Word2Vec/word2vec_model.txt'
DATA_PATH = "../../Fake News Dataset/Train_Test/TrainTest.csv"
STOP_WORD = '../../Data/StopWords'
ALEXA_RANK = '../../Data/AlexaRank'
POS_PATH = "../../Data/POS/pos.csv"

API = "../../API/Models/"

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300
epochs = 3

in_channels = 1
out_channels = 256
kernel_heights = [1, 2, 3, 4]
stride = 1
padding = 0
keep_probab = 0.8