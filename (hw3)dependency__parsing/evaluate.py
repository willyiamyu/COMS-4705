import tensorflow as tf
tf.compat.v1.disable_eager_execution() 
from decoder import Parser
from extract_training_data import FeatureExtractor
from conll_reader import conll_reader
import sys


def compare_parser(target, predict):
    target_unlabeled = set((d.id,d.head) for d in target.deprels.values())
    target_labeled = set((d.id,d.head,d.deprel) for d in target.deprels.values())
    predict_unlabeled = set((d.id,d.head) for d in predict.deprels.values())
    predict_labeled = set((d.id,d.head,d.deprel) for d in predict.deprels.values())

    labeled_correct = len(predict_labeled.intersection(target_labeled))
    unlabeled_correct = len(predict_unlabeled.intersection(target_unlabeled))
    num_words = len(predict_labeled)
    return labeled_correct, unlabeled_correct, num_words 


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    total_labeled_correct = 0
    total_unlabeled_correct = 0
    total_words = 0

    las_list = []
    uas_list = []    
   
    count = 0 
    with open(sys.argv[2],'r') as in_file: 
        print("Evaluating. (Each . represents 100 test dependency trees)")
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            predict = parser.parse_sentence(words, pos)
            labeled_correct, unlabeled_correct, num_words = compare_parser(dtree, predict)
            las_s = labeled_correct / float(num_words)
            uas_s = unlabeled_correct / float(num_words)
            las_list.append(las_s)
            uas_list.append(uas_s)
            total_labeled_correct += labeled_correct
            total_unlabeled_correct += unlabeled_correct
            total_words += num_words
            count +=1 
            if count % 100 == 0:
                print(".",end="")
                sys.stdout.flush()
    print()

    las_micro = total_labeled_correct / float(total_words)
    uas_micro = total_unlabeled_correct / float(total_words)

    las_macro = sum(las_list) / len(las_list)
    uas_macro = sum(uas_list) / len(uas_list)

    print("{} sentence.\n".format(len(las_list)))
    print("Micro Avg. Labeled Attachment Score: {}".format(las_micro))
    print("Micro Avg. Unlabeled Attachment Score: {}\n".format(uas_micro))
    print("Macro Avg. Labeled Attachment Score: {}".format(las_macro))
    print("Macro Avg. Unlabeled Attachment Score: {}".format(uas_macro))
