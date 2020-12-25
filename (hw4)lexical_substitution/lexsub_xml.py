#!/usr/bin/env python
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring
import sys
import re
import codecs

class Context(object):
    """
    Represent a single input word with context.
    """
    
    def __init__(self, cid, word_form, lemma, pos, left_context, right_context): 
        self.cid = cid
        self.word_form = word_form
        self.lemma = lemma
        self.pos = pos
        self.left_context = left_context
        self.right_context = right_context

    def __repr__(self):
        return "<Context_{cid}/{lemma}.{pos} {left} *{word}* {right}>".format(cid=self.cid, lemma = self.lemma, pos = self.pos, left = " ".join(self.left_context), word=self.word_form, right=" ".join(self.right_context))

class LexsubData(object):

    def __init__(self):
        self.total_count =  1
        pass

    def process_context(self, context_s):
        head_re = re.compile("<head>(.*)</head>")
        match =  head_re.search(context_s)
        target = match.groups(1)[0]
        context_left = context_s[:match.start()]
        context_right = context_s[match.end():]
        return target, context_left.split(), context_right.split()

    def parse_lexelt(self, lexelt):
        lex_item = lexelt.get('item')
        parts = lex_item.split('.')
        if len(parts) == 3:
            lemma, pos = parts[0], parts[2]
        else: 
            lemma, pos = parts[0], parts[1]

        for instance in lexelt:
            assert instance.tag=="instance"
            context = instance.find("context")                     
            context_s = "".join([str(context.text)] + [codecs.decode(ET.tostring(e),"UTF-8") for e in context])
            word_form, left_context, right_context = self.process_context(context_s)
            yield Context(self.total_count, word_form, lemma, pos, left_context, right_context)
            self.total_count += 1

    def parse_et(self,et):
       assert et.tag == "corpus"
       for lexelt in et: 
            assert lexelt.tag == "lexelt"
            for annotation in self.parse_lexelt(lexelt):
                yield annotation


def read_lexsub_xml(*sources):
    """
    Parse the lexical substitution data and return an iterator over Context instances.
    """
    lexsub_data = LexsubData()
    for source_f in sources:
        et = ET.parse(source_f)
        for annotation in lexsub_data.parse_et(et.getroot()):
            yield annotation
    
if __name__=="__main__":

    for context in read_lexsub_xml(sys.argv[1]):
        print(context)

