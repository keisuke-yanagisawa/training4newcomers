import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def check_args(args):
    if args.train is None:
        print(parser.print_help())
        exit(1)
    if args.test is not None and args.out is None:
        print(parser.print_help())
        exit(1)

def generate_input(df, window_radius=1):
    _data = []
    for _, item in df.iterrows():
        seq = item.sequence
        length = len(seq)
        
        seq = ("_" * window_radius) + seq + ("_" * window_radius) #add spacer
        for resn in range(length):
            _in = list(seq[resn:resn+window_radius*2+1])
            _data.append(_in)
    return _data

def generate_label(df):
    label = []
    for _, item in df.iterrows():
        ss = item.label
        for resn, _label in enumerate(ss):
            label.append(int(_label))
    return label

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="example program")
    parser.add_argument("-train", help="path to training data (required)")
    parser.add_argument("-test", help="path to test data (optional)")
    parser.add_argument("-out", help="path to predicted information for test data (required only if --test is set)")
    parser.add_argument("--window_radius", type=int, default=1)
    args = parser.parse_args()

    check_args(args)
    
    train_df = pd.read_csv(args.train)

    _data = []
    label = []

    window_radius = args.window_radius
    _data = generate_input(train_df, window_radius)
    label = generate_label(train_df)

    transformer = OneHotEncoder().fit(_data)
    data = transformer.transform(_data)
    label = np.array(label)


    # data splitting
    X_train, X_val, y_train, y_val = train_test_split(data, label, random_state=0)

    # training
    model = LogisticRegression().fit(X_train, y_train)
    
    # prediction & evaluation
    score = model.score(X_val, y_val)
    print('Q2 accuracy: %.4f'%(score))
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print('AUC: %.4f'%(auc))


    if args.test is not None:
        test_df = pd.read_csv(args.test)
        test_data_ = generate_input(test_df, window_radius)
        test_data = transformer.transform(test_data_)
        predicted = model.predict_proba(test_data)[:, 1]

        sequence_id_list    = []
        residue_number_list = []
        for _, item in test_df.iterrows():
            sequence_id = item.sequence_id
            sequence    = item.sequence
            for i, aa in enumerate(sequence):
                sequence_id_list.append(sequence_id)
                residue_number_list.append(i+1) #0-origin to 1-origin

        predicted_df = pd.DataFrame.from_dict({
            "sequence_id": sequence_id_list,
            "residue_number": residue_number_list,
            "predicted_value": predicted,
            })
        predicted_df.to_csv(args.out, index=None)
            
