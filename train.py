import argparse
import logging
import pdb
import sys
from os import listdir
import traceback
import pickle
import predictor
import dataset

def main(args):

    # training mode
    if args.train:
        logging.info("start training...")
        EG_predictor = predictor.Predictor(max_epochs=args.epoch)

        # load data and create a data set
        with open("./data/train.pkl", "rb") as f:
            train = pickle.load(f)

        logging.info("create a dataset...")
        train_dataset = dataset.Dataset(data=train)
        logging.info("start training!")
        EG_predictor.fit_dataset(data=train_dataset, collate_fn=train_dataset.collate_fn)

    # test mode
    if args.test:
        logging.info("start testing...")
        EG_predictor = predictor.Predictor(max_epochs=args.epoch)

        # choose the model to test data
        EG_predictor.load("model-40")

        # load data and create a data set
        with open("./data/test.pkl", 'rb') as f:
            test = pickle.load(f)

        logging.info("create a dataset...")
        test_dataset = dataset.Dataset(data=test)
        pre, ans = EG_predictor.predict_dataset(data=test_dataset, collate_fn=test_dataset.collate_fn)

        # calculate acc
        correct = 0
        for i in zip(pre, ans):
            pre_ans = 0
            # class number is 7
            for j in range(1, 7):
                if i[0][j] > i[0][pre_ans]:
                    pre_ans = j
            if pre_ans == i[1]:
                correct += 1

        print("correct: ", correct)
        print("acc: ", correct/len(ans))

    # label the raw ecg data
    if args.label:
        logging.info("start labeling...")
        EG_predictor = predictor.Predictor(max_epochs=args.epoch)
        EG_predictor.load("model-40")

        # load data and create a dataset
        path = "./ecg/"
        files = listdir(path)
        for file in files:
            if ".pkl" in file:
                with open(path + file, 'rb') as f:
                    test = pickle.load(f)
                    logging.info("create a dataset...")
                    test_dataset = dataset.Dataset(data=test)
                    pre, _ = EG_predictor.predict_dataset(data=test_dataset, collate_fn=test_dataset.collate_fn)

                    with open("./pre/" + file.split('.')[0] + ".pkl", "wb") as wf:
                        pickle.dump(pre, wf)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--data_dir', default="./data/", type=str,
                        help='Directory to training data')
    parser.add_argument('--model', default="single", type=str,
                        help='Which model to use')
    parser.add_argument('--epoch', default=41, type=int,
                        help='How many epoch to train the model')
    parser.add_argument('--test', action="store_true", default=False,
                        help='Test the model')
    parser.add_argument('--train', action="store_true", default=False,
                        help='Train the model')
    parser.add_argument('--label', action="store_true", default=False,
                        help='label the raw datas')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
