from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

win_radius = 25

n = 0
_data = []
label = []

for line in open("train.dat").readlines():
    if n%2 == 0:
        seq = line.rstrip("\n")            
    else:
        ss  = line.rstrip("\n")

        m = 0
        seq = ("_" * win_radius) + seq + ("_" * win_radius) #add spacer
        for _label in ss:
            label.append(int(_label))
            _in = list(seq[m:m+win_radius*2+1])
            _data.append(_in)
            m += 1

    n += 1

transformer = OneHotEncoder().fit(_data)
dst = transformer.transform(_data)
data = dst.toarray()
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
score = model.score(X_test, y_test)
print('Q2 accuracy: %.4f'%(score))

auc = roc_auc_score(y_test, model.predict(X_test))
print('AUC: %.4f'%(auc))
