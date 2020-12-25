"""
COMS W4705 - Natural Language Processing - Fall 20 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum, isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        # check if lhs items sum to one
        for key, val in self.lhs_to_rules.items():
            prob = 0
            for i in val:
                # sum up all probablities for non-terminal on LHS
                prob += i[-1]

                # check if in CNF
                if len(i[1]) == 1:
                    # instances where terminal contains no letters, only punctuation
                    # assume these 'only punctuation' strings are terminals
                    if i[1][0].isalpha():
                        # check if string is in lower case for terminasl
                        if not i[1][0].islower():
                            print('Grammar is not in CNF: check production rules (terminals)')
                            return False

                elif len(i[1]) == 2:
                    # check if both elements in tuple are uppercase/nonterminal
                    if (i[1][0].isupper() and i[1][1].isupper()):
                        pass
                    else:
                        print('Grammar is not in CNF: check production rules (nonterminals)')
                        return False
                else:
                    print('Grammar is not in CNF: one or more production rules leads to > 2 terms')
                    return False

            # check if probabilities sum to 1
            if not isclose(1, prob):
                print('Grammar is not valid: LHS probabilities do not sum to 1')
                return False
        print('Grammar is valid: in CNF and LHS probabilities sum to 1')
        return True 


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        
