from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            feature = self.extractor.get_input_representation(words, pos, state)
            pred = self.model.predict(np.array([feature,]))
            indices = np.argsort(-pred, axis=1)[0]
            for i in indices:
                key, val = self.output_labels[i]
                if key == 'shift':
                    if len(state.buffer) == 1:
                        if len(state.stack) == 0:
                            state.shift()
                            break
                    elif len(state.buffer) > 1:
                            state.shift()
                            break
                elif key == 'left_arc':
                    if len(state.stack) > 0:
                        if val != 'root':
                            state.left_arc(val)
                            break
                elif key == 'right_arc':
                    if len(state.stack) != 0:
                        state.right_arc(val)
                        break
            # TODO: Write the body of this loop for part 4 

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

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

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
