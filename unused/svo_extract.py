# Copyright 2017 Peter de Vocht
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import en_core_web_sm
from collections.abc import Iterable

# use spacy small model
nlp = en_core_web_sm.load()

# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}


# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet):
    return "and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights} # 查找si的右子单词并小写化为rightDeps
        if contains_conj(rightDeps): # 若rightDeps包含["and", "or", "nor", "but", "yet", "so", "for"]:
            # 则从si的右子单词中查找在SUBJECTS列表的单词或名词；
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            # 若从si的右子单词中查找在SUBJECTS列表的单词或名词后:
            if len(more_subs) > 0:
                # 则递归执行d1.3.1~d1.3.3操作
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs: # 遍历每个宾语oi
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights} # 查找oi的右子单词并小写化为rightDeps；
        if contains_conj(rightDeps): # 若rightDeps包含["and", "or", "nor", "but", "yet", "so", "for"]
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"]) # 则从oi的右子单词中查找是否为宾语的单词或名词；
            if len(more_objs) > 0: # 若从oi的右子单词中查找有为宾语的单词或名词后:
                more_objs.extend(_get_objs_from_conjunctions(more_objs)) # 则递归执行d3.4.4.1~d3.4.4.3操作
    return more_objs


# find sub dependencies
def _find_subs(tok):
    # 查找vi作为动词、名词和根单词的父单词，若非则沿树往上找父单词的父单词，直到满足条件为止；
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    # 若查找到作为动词的父单词hi：
    if head.pos_ == "VERB":
        # subs = [tok for tok in head.lefts if tok.dep_ == "SUB"] # Exist Error
        # 遍历hi的左子单词，查找在SUBJECTS列表且POS标签非限定词的单词，添加进主语列表；
        subs = [tok for tok in head.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
        if len(subs) > 0: # 若查找到主语:  
            verb_negated = _is_negated(head) # 则判断父单词hi是否为否定词
            subs.extend(_get_subs_from_conjunctions(subs)) # 然后重复d1.3操作
            return subs, verb_negated # 返回主语列表和动词否定标记；
        elif head.head != head: # 若未查找到主语，则判断父单词hi是否为根单词，然后重复d1.4操作；
            return _find_subs(head)
    elif head.pos_ == "NOUN":  # 若查找到作为名词的父单词hi
        return [head], _is_negated(tok) # 则返回当前父单词hi，以及vi的否定词标记：
    return [], False # 递归遍历后仍未找到，则返回[]，以及False；


# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights) # 等价于list(tok.children)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False


# get all the verbs on tokens with negation marker
def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps: # 遍历vi的右子单词，查找POS标签为介词+(依赖标签为介词修饰符/被动句+行为主体)的单词ri:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            # 遍历ri的右子单词查找在OBJECTS列表，或POS标签为代词的me，或被动句+介词宾语的单词，添加为宾语；
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])  # 德语第4格(宾格): tok.lower_ == "mich"
    return objs


# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# xcomp; open complement - verb has no subject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps: # 遍历vi的右子单词
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp": # 查找POS标签为动词且依赖标签为开放子句补语的单词ri
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS] # 遍历ri的右子单词查找在OBJECTS列表的单词，添加为宾语
            objs.extend(_get_objs_from_prepositions(rights, is_pas))  # 再从介词中扩展宾语(执行d2.4.2操作)
            if len(objs) > 0: # 若从开放子句补语中查找出宾语
                return v, objs # 则返回ri以及宾语列表
    return None, None  # 否则返回None，None；


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    # 判断vi是否为否定词：即其子单词的小写化是否在NEGATIONS列表；
    verb_negated = _is_negated(v)
    # 遍历vi的左子单词，查找在SUBJECTS列表且POS标签非限定词的单词，添加进主语列表；
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0: # 若查找到主语，则遍历每个主语si，从并列连词中扩展主语：
        subs.extend(_get_subs_from_conjunctions(subs)) # 将查找到的主语添加到subs中，与vi的否定词标记一起返回；
    else:  # 若未查找到主语：
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs) # 将查找到的主语添加到subs中，与vi的否定词标记一起返回；
    return subs, verb_negated


# find the main verb - or any aux verb if we can't find it
def _find_verbs(tokens): # 查找主要动词：
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)] # 首先排除助动词，查找“VERB”；
    if len(verbs) == 0: # 若查找不到，则查找含助动词（主、被动）的动词；
        verbs = [tok for tok in tokens if _is_verb(tok)]
    return verbs


# is the token a verb?  (excluding auxiliary verbs)
def _is_non_aux_verb(tok): # 排除助动词，查找“VERB”；
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass")


# # is the token a verb?  (excluding auxiliary verbs)
def _is_verb(tok): # 查找含助动词（主、被动）的动词；
    return tok.pos_ == "VERB" or tok.pos_ == "AUX"

# is the token a verb?  (excluding auxiliary verbs)
# def _is_verb(tok): # 查找含助动词（主、被动）的动词；
#     return tok.pos_ == "VERB" or tok.dep_ == "aux" or tok.dep_ == "auxpass"

# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ': # 判断vi的右子单词是否为2个以上，且第一个为并列连词；
        for tok in rights[1:]:
            if _is_non_aux_verb(tok): # # 则判断第2个开始的单词是否为动词，排除助动词；
                return True, tok # 是则返回True和修改后的动词，否则返回False，当前动词vi;

    return False, v


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)
    # 遍历vi的右子单词，查找在OBJECTS列表，或被动句中作为介词宾语的单词为宾语；
    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')] #
    objs.extend(_get_objs_from_prepositions(rights, is_pas))  # 从介词中扩展宾语

    #potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas) # 从开放子句补语中扩展宾语：
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0: # 若从开放子句补语中查找出宾语
        objs.extend(potential_new_objs) # 则扩展宾语列表，且主要动词替换为ri；
        v = potential_new_verb
    if len(objs) > 0: # 若宾语列表不为空，从并列连词中扩展宾语： 
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs  # 返回主要动词(可能经开放子句补语修改过)和扩展后的宾语列表；


# return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return None


# simple stemmer using lemmas
def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


# print information for displaying all kinds of things of the parse tree
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


# expand an obj / subj np using its chunk
def expand(item, tokens, visited):
    if item.lower_ == 'that':
        temp_item = _get_that_resolution(tokens)
        if temp_item is not None:
            item = temp_item

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN":
                if item2.i not in visited:
                    visited.add(item2.i)
                    parts.extend(expand(item2, tokens, visited))
            break

    return parts


# convert a list of tokens to a string
def to_str(tokens):
    if isinstance(tokens, Iterable):
        return ' '.join([item.text for item in tokens])
    else:
        return ''


# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens):
    svos = []
    # 判断是否为被动句：有无依赖项为"auxpass"的被动助动词；
    is_pas = _is_passive(tokens)
    # 查找主要动词：首先排除助动词，查找“VERB”；若查找不到，则查找含助动词（主、被动）的动词；
    verbs = _find_verbs(tokens)
    # 创建一个visited集合，记录已查找过的单词；
    visited = set()  # recursion detection
    for v in verbs: # 遍历主要动词vi：
        subs, verbNegated = _get_all_subs(v) # 根据vi获得其临近的主语si:
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0: # 判断动词vi临近的主语si是否为空，为空则不再检查vi：
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb: # 若动词vi被识别为并列动词
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expand(obj, tokens, visited)),
                                         "!" + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                            svos.append((to_str(expand(obj, tokens, visited)),
                                         "!" + v2.lemma_ if verbNegated or objNegated else v2.lemma_, to_str(expand(sub, tokens, visited))))
                        else:
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "!" + v.lower_ if verbNegated or objNegated else v.lower_, to_str(expand(obj, tokens, visited))))
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "!" + v2.lower_ if verbNegated or objNegated else v2.lower_, to_str(expand(obj, tokens, visited))))
            else:     # 若动词vi未被识别为并列动词
                v, objs = _get_all_objs(v, is_pas) # 根据vi和被动句标记查找其宾语：
                for sub in subs: # 遍历主语列表中每个主语si:
                    if len(objs) > 0: #  若宾语列表不为空：
                        for obj in objs: # 遍历每个宾语oi
                            objNegated = _is_negated(obj) # 判断oi是否为否定词；
                            if is_pas:  # reverse object / subject for passive # 若为被动句，则宾谓主：
                                svos.append((to_str(expand(obj, tokens, visited)),
                                             "!" + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                            else: # d3.1.4  若为主动句，则为主谓宾：与d3.1.2操作一致，只是先由块扩展subj/obj；
                                svo_subjs = expand(sub, tokens, visited)  # 真正的主语(token类型)
                                svo_verbs = [v]  # 真正的谓语(token类型)
                                svo_objs = expand(obj, tokens, visited)  # 真正的宾语(token类型)

                                # 取出谓语中全部token的索引号，然后在取出宾语中token的索引号
                                # 然后互换位置,并按照新的索引顺序打印出句子
                                verbs_i = [tok.i for tok in svo_verbs] #if isinstance(svo_verbs, list) else svo_verbs.i
                                objs_i = [tok.i for tok in svo_objs] #if isinstance(svo_objs, list) else svo_objs.i


                                print()
                                remain_tokens = [tok for tok in tokens if tok not in svo_subjs+svo_verbs+svo_objs]
                                ordered_sent = []

                                # for tok in tokens:
                                #     if tok not in svo_subjs+svo_verbs+svo_objs:
                                #         return True

                                # svos.append((to_str(expand(sub, tokens, visited)),
                                #              "!" + v.lower_ if verbNegated or objNegated else v.lower_, to_str(expand(obj, tokens, visited))))
                    else:
                        # no obj - just return the SV parts  # 主谓由于没有宾语，无需改变

                        sovs = tokens.text
                        print(sovs)
                        # svos.append((to_str(expand(sub, tokens, visited)),
                        #              "!" + v.lower_ if verbNegated else v.lower_,))

    return svos