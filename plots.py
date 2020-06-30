import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

n_classes = 8
classes = ['IfcBeam', 'IfcColumn', 'IfcCovering', 'IfcDoor', 'IfcRailing',
       'IfcSlab', 'IfcWallStandardCase', 'IfcWindow']

colors = ['darkorange', 'cornflowerblue', 'purple', 'gold','darkred','lightcoral','tomato','slateblue' ]

model = 'graph'
labels = pd.read_csv('./results/'+model+'/labels.csv', header=0, index_col=0)
scores = pd.read_csv('./results/'+model+'/prob.csv',header=0, index_col=0)

y_true = labels['y_true'].values
y_pred = labels['y_pred'].values

print(metrics.classification_report(y_pred, y_true))

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

onehot_y = encode_onehot(y_true)
onehot_pred = encode_onehot(y_pred)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(onehot_y[:, i],
                                                        scores.values[:, i])
    average_precision[i] = average_precision_score(onehot_y[:, i], scores.values[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(onehot_y.ravel(),scores.values.ravel())
average_precision["micro"] = average_precision_score(onehot_y, scores.values,average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


plt.figure(figsize=(7, 7))
f_scores = np.linspace(0.2, 0.8, num=10)
lines = []
labels = []

linestyles = cycle(['-', '--', '-.', ':','-', '--', '-.', ':', '--', '--'])

for i, line in zip(range(n_classes), linestyles):

    l, = plt.plot(recall[i], precision[i], color=colors[i], linestyle=line, lw=2)
    lines.append(l)
    labels.append('{0} (area = {1:0.2f})'
                  ''.format(classes[i], average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
# plt.legend(lines, labels, loc=(0.05, 0.3), prop=dict(size=10))
plt.legend(lines, labels, prop=dict(size=12))
plt.tight_layout()
plt.savefig('./results/'+model+'/plot_prc.png', dpi=300)