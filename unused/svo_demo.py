import unittest

from nltk import Tree

from unused.svo_extract import findSVOs, printDeps, nlp


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


# tok = nlp("expert spacy users are very kind to dogs")
# print([to_nltk_tree(sent.root).pretty_print() for sent in tok.sents])
# svos = findSVOs(tok)
# # printDeps(tok)
# print(svos)


tok = nlp("both sides should understand that")
print([to_nltk_tree(sent.root).pretty_print() for sent in tok.sents])
svos = findSVOs(tok)
# printDeps(tok)
print(svos)


str1 = "Then there’s a development setback on top of that that pushes you even further back."
str2 = "And that goes with that we’re going to do things differently, but we haven’t done that yet."
str3 = "Seated in Mission Control, Chris Kraft neared the end of a tedious Friday afternoon as he monitored a " \
       "seemingly interminable ground test of the Apollo 1 spacecraft."

tokens1 = nlp(str1)
print([to_nltk_tree(sent.root).pretty_print() for sent in tokens1.sents])
svos1 = findSVOs(tokens1)
print("\n1")
print(str1)
print(svos1)

tokens2 = nlp(str2)
svos2 = findSVOs(tokens2)
print("\n2")
print(str2)
print(svos2)

tokens3 = nlp(str3)
svos3 = findSVOs(tokens3)
print("\n3")
print(str3)
print(svos3)



# test the subject/verb/object_extraction
class SubjectVerbOjectExtractTest(unittest.TestCase):
    def __init__(self, methodName: str):
        unittest.TestCase.__init__(self, methodName)

    # test
    def test_svo_1(self):
        tok = nlp("the annoying person that was my boyfriend hit me")
        printDeps(tok)  # just show what printDeps() does
        print([to_nltk_tree(sent.root).pretty_print() for sent in tok.sents])
        svos = findSVOs(tok)
        self.assertTrue(set(svos) == {('the annoying person', 'was', 'my boyfriend'), ('the annoying person', 'hit', 'me')})

    def test_svo_2(self):
        tok = nlp("making $12 an hour? where am i going to go? I have no other financial assistance available and he certainly won't provide support.")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('I', '!have', 'other financial assistance available'), ('he', '!provide', 'support')})

    def test_svo_3(self):
        tok = nlp("I don't have other assistance")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('I', '!have', 'other assistance')})

    def test_svo_4(self):
        tok = nlp("They ate the pizza with anchovies.")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('They', 'ate', 'the pizza')})

    def test_svo_5(self):
        tok = nlp("I have no other financial assistance available and he certainly won't provide support.")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('I', '!have', 'other financial assistance available'), ('he', '!provide', 'support')})

    def test_svo_6(self):
        tok = nlp("I have no other financial assistance available, and he certainly won't provide support.")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('I', '!have', 'other financial assistance available'),
                                      ('he', '!provide', 'support')})

    def test_svo_7(self):
        tok = nlp("he did not kill me")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', '!kill', 'me')})

    def test_svo_8(self):
        tok = nlp("he is an evil man that hurt my child and sister")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', 'is', 'an evil man'),
                                      ('an evil man', 'hurt', 'my child'),
                                      ('an evil man', 'hurt', 'sister')})

    def test_svo_9(self):
        tok = nlp("he told me i would die alone with nothing but my career someday")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', 'told', 'me')})

    def test_svo_10(self):
        tok = nlp("I wanted to kill him with a hammer.")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('I', 'kill', 'him')})

    def test_svo_11(self):
        tok = nlp("because he hit me and also made me so angry I wanted to kill him with a hammer.")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', 'hit', 'me'), ('I', 'kill', 'him')})

    def test_svo_12(self):
        tok = nlp("he and his brother shot me")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', 'shot', 'me'), ('his brother', 'shot', 'me')})

    def test_svo_13(self):
        tok = nlp("he and his brother shot me and my sister")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', 'shot', 'me'), ('he', 'shot', 'my sister'),
                                      ('his brother', 'shot', 'me'), ('his brother', 'shot', 'my sister')})

    def test_svo_14(self):
        tok = nlp("the boy raced the girl who had a hat that had spots.")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('the boy', 'raced', 'the girl'), ('who', 'had', 'a hat'),
                                      ('a hat', 'had', 'spots')})

    def test_svo_15(self):
        tok = nlp("he spit on me")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', 'spit', 'me')})

    def test_svo_16(self):
        tok = nlp("he didn't spit on me")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', '!spit', 'me')})

    def test_svo_17(self):
        tok = nlp("the boy raced the girl who had a hat that didn't have spots.")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('the boy', 'raced', 'the girl'), ('who', 'had', 'a hat'),
                                      ('a hat', '!have', 'spots')})

    def test_svo_18(self):
        tok = nlp("he is a nice man that didn't hurt my child and sister")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', 'is', 'a nice man'), ('a nice man', '!hurt', 'my child'),
                                      ('a nice man', '!hurt', 'sister')})

    def test_svo_19(self):
        tok = nlp("he didn't spit on me and my child")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', '!spit', 'me'), ('he', '!spit', 'my child')})

    def test_svo_20(self):
        tok = nlp("he didn't spit on me or my child")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', '!spit', 'me'), ('he', '!spit', 'my child')})

    def test_svo_21(self):
        tok = nlp("he didn't spit on me nor my child")
        svos = findSVOs(tok)
        # printDeps(tok)
        self.assertTrue(set(svos) == {('he', '!spit', 'me'), ('he', '!spit', 'my child')})

    def test_svo_22(self):
        tok = nlp("he beat and hurt me")
        # printDeps(tok)
        svos = findSVOs(tok)
        self.assertTrue(set(svos) == {('he', 'beat', 'me'), ('he', 'hurt', 'me')})

    def test_svo_23(self):
        tok = nlp("I was beaten by him")
        # printDeps(tok)
        print([to_nltk_tree(sent.root).pretty_print() for sent in tok.sents])
        svos = findSVOs(tok)
        self.assertTrue(set(svos) == {('him', 'beat', 'I')})

    def test_svo_24(self):
        tok = nlp("lessons were taken by me")
        # printDeps(tok)
        svos = findSVOs(tok)
        self.assertTrue(set(svos) == {('me', 'take', 'lessons')})

    def test_svo_25(self):
        tok = nlp("Seated in Mission Control, Chris Kraft neared the end of a tedious Friday afternoon as he monitored a seemingly interminable ground test of the Apollo 1 spacecraft.")
        # printDeps(tok)
        svos = findSVOs(tok)
        self.assertTrue(set(svos) == {('Chris Kraft', 'neared', 'the end of a tedious Friday afternoon'), ('he', 'monitored', 'a interminable ground test of the Apollo spacecraft')})

Test = SubjectVerbOjectExtractTest('test_svo_23')
Test.test_svo_1()
Test.test_svo_2()
Test.test_svo_3()
Test.test_svo_4()
Test.test_svo_5()
Test.test_svo_6()
Test.test_svo_7()
Test.test_svo_8()
Test.test_svo_9()
Test.test_svo_10()
Test.test_svo_11()
Test.test_svo_12()
Test.test_svo_13()
Test.test_svo_14()
Test.test_svo_15()
Test.test_svo_16()
Test.test_svo_17()
Test.test_svo_18()
Test.test_svo_19()
Test.test_svo_20()
Test.test_svo_21()
Test.test_svo_22()
Test.test_svo_23()
Test.test_svo_24()
Test.test_svo_25()


