from cky import Pcfg, CkyParser, get_tree
import sys

def tokenize(line):
    tok = ''
    for c in line: 
        if c == " ":
            if tok: 
                yield tok
                tok = ""
        elif c == "(" or c==")":
            if tok: 
                yield tok
            yield c
            tok = ""
        else: 
            tok += c
    if tok: 
        yield tok
        tok = ""
           
def parse_tree(line):
    toks = tokenize(line)
    stack = []
    t = next(toks)
    try:
        while t:
            if t=="(":
                stack.append(t)
            elif t==")":
                subtree = []
                s = stack.pop()
                while s[0]!="(":
                    subtree.append(s)
                    s = stack.pop()
                stack.append(tuple(reversed(subtree)))
            else: 
                stack.append(t)
            t = next(toks)
    except StopIteration: 
        return stack.pop()
                

def get_leafs(tree):
    if isinstance(tree,str):
        return [tree]
    else: 
        result = []
        for x in tree[1:]:
            result.extend(get_leafs(x))
        return result
            

def get_constituents(tree,left=0):
    if not tree: 
        return [], left
    start = left
    if isinstance(tree,str): 
        return [],left+1
    else: 
        result = []
        phrase = tree[0]
        for subtree in tree[1:]:
            subspans, right = get_constituents(subtree, left)
            result.extend(subspans)
            left = right
        result.append((phrase,start,left))
        return result, left

def compute_parseval_scores(gold_tree, test_tree): 
    
    gold_const = set(get_constituents(gold_tree)[0])
    test_const = set(get_constituents(test_tree)[0])
    
    if not test_const: 
        return 0.0,0.0,0.0

    correct = len(gold_const.intersection(test_const))     
    recall = correct / float(len(gold_const))
    precision = correct / float(len(test_const))
    fscore = (2*precision*recall) / (precision+recall)
    return precision, recall, fscore 

def evaluate_parser(parser, treebank_file):
  
    total = 0
    unparsed = 0
    fscore_sum = 0.0
    for line in treebank_file:  
        gold_tree = parse_tree(line.strip())
        tokens = get_leafs(gold_tree)
        print("input: ",tokens)
        chart,probs = parser.parse_with_backpointers(tokens)
        print("target:    ",gold_tree)
        total += 1
        if not chart: 
            unparsed += 1
            res = tuple()
        else: 
            try:
                res = get_tree(chart,0,len(tokens),parser.grammar.startsymbol)
            except KeyError:
                unparsed += 1
                res = tuple() 
        print("predicted: ",res)
        #print(compute_parseval_scores(gold_tree, res))
        p,r,f = compute_parseval_scores(gold_tree, res)
        fscore_sum += f
        print("P:{} R:{} F:{}".format(p,r,f))
        print()
        
    parsed = total-unparsed 
    if parsed == 0:
        coverage = 0.0
        fscore_parsed = 0.0
        fscore_all = 0.0 
    else: 
        coverage =  (parsed / total) *100
        fscore_parsed = fscore_sum / parsed 
        fscore_all = fscore_sum / total
    print("Coverage: {:.2f}%, Average F-score (parsed sentences): {}, Average F-score (all sentences): {}".format(coverage, fscore_parsed, fscore_all))
        

if __name__ == "__main__":

    if len(sys.argv)!=3:
        print("USAGE: python evaluate_parser.py [grammar_file] [test_file]")
        sys.exit(1)

    with open(sys.argv[1],'r') as grammar_file, open(sys.argv[2],'r') as test_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        evaluate_parser(parser,test_file)
