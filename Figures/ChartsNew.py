import numpy as np
import matplotlib.pyplot as plt

# Values of each group


def plotSVMvsLR(figure_name):

    bars1 = [81, 90, 86, 85, 49, 49, 85, 89]
    bars2 = [23, 50, 32, 52, 2, 51, 52, 63]
    barWidth = 0.3
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, bars1, color='#ec7063', width=barWidth, edgecolor='white')
    plt.bar(r2, bars2, color='#85c1e9', width=barWidth, edgecolor='white')

    # Add xticks on the middle of the group bars
    plt.ylabel('F1', fontweight='bold')
    plt.xticks([r + (barWidth/2) for r in range(len(bars1))], ['Unigram','C3-Gram','C3+C4+C5','L', 'Embedding(F)', 'Embedding(N)','L+E','AllFeatures']
               , rotation=45, horizontalalignment='right', fontweight='bold')

    # Create legend & Show graphic
    plt.legend(['SVM', 'LR'], loc='center left',bbox_to_anchor=(1, 0.5))
    plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.25, alpha=0.7)
    # plt.spines['top'].set_visible(False)
    # plt.show()
    fig = plt.figure(1)
    fig.savefig(figure_name, bbox_inches='tight')


def plot(figure_name, bars1, exp):


    barWidth = 0.6
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))

    plt.bar(r1, bars1, color='#85c1e9', width=barWidth, edgecolor='white')

    # Add xticks on the middle of the group bars
    plt.ylabel('F1', fontweight='bold')
    plt.xticks([r  for r in range(len(bars1))], exp
               , fontweight='bold')

    # Create legend & Show graphic
    plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.25, alpha=0.7)
    # plt.spines['top'].set_visible(False)
    # plt.show()
    plt.savefig(figure_name, bbox_inches='tight')


# human_scores = [44, 42, 14]
# options = ['The content is unrealistic', 'Has no trustworthy source', 'others']
# plot("human-Fake-Options.png", human_scores, options)
# human_scores = [62, 21, 17]
# options = ['The content is believable', 'Source is Reliable', 'others']
# plot("human-True-Options.png", human_scores, options)
#
# human_scores = [58, 65, 70, 68, 63]
# options = ['P1', 'P2', 'P3', 'P4', 'P5']
# plot("human-F1.png", human_scores, options)

# nn_scores = [62, 22, 52, 45, 57, 63, 75]
# exp_name = ['CNN', 'RNN', 'RCNN', 'LSTM', 'LSTM Attention', 'BiLSTM Attention', 'BERT']
# plotNN("NN.png",nn_scores, exp_name)

plotSVMvsLR("LRvsSVM")