import json
import pdb
from BuildKG import Map2doc
from relation_metrics import span_metric
def ReadJson(senfn):
    docs = {}
    with open(senfn) as f:
        docs_sent = [json.loads(jsonline) for jsonline in f.readlines()]
    for i in range(len(docs_sent)):
        docs[docs_sent[i]['doc_key']] = docs_sent[i]
        if 'relations' in docs[docs_sent[i]['doc_key']]:
            docs[docs_sent[i]['doc_key']]['relation'] = flat(docs[docs_sent[i]['doc_key']]['relations'])
        docs[docs_sent[i]['doc_key']]['ner'] = GetNER(docs[docs_sent[i]['doc_key']]['ner'])
        # Map2doc(docs[docs_sent[i]['doc_key']])
    return docs
def GetNER(lst):
    NERdir = {}
    for sent in lst:
        for ent in sent:
            NERdir[tuple(ent[:2])] = ent[-1]
    return NERdir
def flat(lst):
    new_lst = []
    for ele in lst:
        new_lst += ele
    return new_lst
def ReadJsonPred(senfn, truedocs):
    docs = {}
    with open(senfn) as f:
        docs_sent = [json.loads(jsonline) for jsonline in f.readlines()]
    for i in range(len(docs_sent)):
        docs[docs_sent[i]['doc_key']] = docs_sent[i]
        if 'relations' in docs[docs_sent[i]['doc_key']]:
            docs[docs_sent[i]['doc_key']]['relation'] = docs[docs_sent[i]['doc_key']]['relations']

        docs[docs_sent[i]['doc_key']]['sentences'] = truedocs[docs_sent[i]['doc_key']]['sentences']
        docs[docs_sent[i]['doc_key']]['ner'] = [[] for shit in range(len(docs[docs_sent[i]['doc_key']]['sentences']))]
        Map2doc(docs[docs_sent[i]['doc_key']])

    return docs

# def Aspect

def GetRelCoref(true_docs, gold_docs,aspect):
    true_rels = []
    for doc_key in true_docs:
        nerdir = gold_docs[doc_key]['ner']
        sentences = true_docs[doc_key]['sentences']
        corefs = gold_docs[doc_key]['clusters']
        coref_set = set()
        for cluster in corefs:
            for span in cluster:
                coref_set.add(tuple(span))
        for relation in true_docs[doc_key]['relation']:

                span0 = tuple(relation[:2])
                span1 = tuple(relation[2:4])
                rel = relation[-1]
                key = False
                phrase1 = ' '.join(sentences[span0[0]:(span0[1]+1)])
                phrase2 = ' '.join(sentences[span1[0]:(span1[1]+1)])
                if span0 in coref_set or span1 in coref_set:
                    key = True
                # if span0 in nerdir and nerdir[span0] == aspect:
                #     key = True
                #     print phrase1
                # if span1 in nerdir and nerdir[span1] == aspect:
                #     key = True
                #     print phrase2
                if key:
                    # pdb.set_trace()
                    # print phrase1 + '\t' + rel + '\t' + phrase2
                    # print phrase1
                    # print phrase2
                    relation_token = [[doc_key + str(span0[0]) + '_' + str(span0[1]) , doc_key + str(span1[0]) + '_' + str(span1[1])], rel]
                    true_rels.append(relation_token)
    return true_rels
                
def GetRel(true_docs, gold_docs,aspect):
    true_rels = []
    for doc_key in true_docs:
        nerdir = gold_docs[doc_key]['ner']
        sentences = true_docs[doc_key]['sentences']
        for relation in true_docs[doc_key]['relation']:

                span0 = tuple(relation[:2])
                span1 = tuple(relation[2:4])
                rel = relation[-1]
                key = False
                phrase1 = ' '.join(sentences[span0[0]:(span0[1]+1)])
                phrase2 = ' '.join(sentences[span1[0]:(span1[1]+1)])

                if span0 in nerdir and nerdir[span0] == aspect:
                    key = True
                    print phrase1
                if span1 in nerdir and nerdir[span1] == aspect:
                    key = True
                    print phrase2
                if key:
                    # pdb.set_trace()
                    # print phrase1 + '\t' + rel + '\t' + phrase2
                    # print phrase1
                    # print phrase2
                    relation_token = [[doc_key + str(span0[0]) + '_' + str(span0[1]) , doc_key + str(span1[0]) + '_' + str(span1[1])], rel]
                    true_rels.append(relation_token)
    return true_rels
def GetRelSpan(true_docs, gold_docs,aspect):
    true_rels = []
    for doc_key in true_docs:
        nerdir = gold_docs[doc_key]['ner']
        sentences = true_docs[doc_key]['sentences']
        for relation in true_docs[doc_key]['relation']:

                span0 = tuple(relation[:2])
                span1 = tuple(relation[2:4])
                rel = relation[-1]
                key = True
                phrase1 = ' '.join(sentences[span0[0]:(span0[1]+1)])
                phrase2 = ' '.join(sentences[span1[0]:(span1[1]+1)])

                if span0 in nerdir and nerdir[span0] == aspect:
                    key = True
                    print phrase1
                if span1 in nerdir and nerdir[span1] == aspect:
                    key = True
                    print phrase2

                
                if doc_key == 'IJCAI_2016_413_abs':
                    # pdb.set_trace()
                    
                    print phrase1 + '\t' + rel + '\t' + phrase2

                    if span0[0] > span1[0]:
                        span0, span1 = span1, span0
                    relation_token = [[doc_key + str(span0[0]) + '_' + str(span0[1]) , doc_key + str(span1[0]) + '_' + str(span1[1])], 'REL']
                    true_rels.append(relation_token)
    return true_rels

def GetRelPhrase(true_docs, gold_docs,aspect, phraseset):
    true_rels = []
    for doc_key in true_docs:
        nerdir = gold_docs[doc_key]['ner']
        sentences = true_docs[doc_key]['sentences']
        for relation in true_docs[doc_key]['relation']:

                span0 = tuple(relation[:2])
                span1 = tuple(relation[2:4])
                rel = relation[-1]
                key = False
                # key = True
                phrase1 = ' '.join(sentences[span0[0]:(span0[1]+1)])
                phrase2 = ' '.join(sentences[span1[0]:(span1[1]+1)])
                # if phrase1 in phraseset or phrase2 in phraseset:
                #     key = True
                if phrase1.isupper():
                    # print phrase1
                    key = True
                if phrase2.isupper():
                    # print phrase2
                    key = True
                # if span0 in nerdir and nerdir[span0] == aspect:
                #     key = True
                #     print phrase1
                # if span1 in nerdir and nerdir[span1] == aspect:
                #     key = True
                #     print phrase2
                if key:
                    # pdb.set_trace()
                    # print phrase1 + '\t' + rel + '\t' + phrase2
                    # print phrase1
                    # print phrase2
                    # print phrase1, phrase2
                    relation_token = [[doc_key + str(span0[0]) + '_' + str(span0[1]) , doc_key + str(span1[0]) + '_' + str(span1[1])], rel]
                    true_rels.append(relation_token)
    return true_rels


def TrueSet(true_docs, gold_docs,aspect):
    true_rels = []
    for doc_key in true_docs:
        nerdir = gold_docs[doc_key]['ner']
        sentences = true_docs[doc_key]['sentences']
        for relation in true_docs[doc_key]['relation']:

                span0 = tuple(relation[:2])
                span1 = tuple(relation[2:4])
                rel = relation[-1]
                key = True
                # key = True
                phrase1 = ' '.join(sentences[span0[0]:(span0[1]+1)])
                phrase2 = ' '.join(sentences[span1[0]:(span1[1]+1)])
                # if phrase1 in phraseset or phrase2 in phraseset:
                #     key = True
                if phrase1.isupper():
                    # print phrase1
                    key = True
                if phrase2.isupper():
                    # print phrase2
                    key = True
                if key:
                    relation_token = doc_key + str(span0[0]) + '_' + str(span0[1]) +'_'+ doc_key + str(span1[0]) + '_' + str(span1[1]) +'_' +  rel
                    true_rels.append(relation_token)
    return true_rels
def PrintError(true_docs, gold_docs,aspect, trueset):
    true_rels = []
    pronoun = ['this', 'it', 'former', 'latter', 'they','those','It','alternative']
    for doc_key in true_docs:
        nerdir = gold_docs[doc_key]['ner']
        sentences = true_docs[doc_key]['sentences']
        for relation in true_docs[doc_key]['relation']:

                span0 = tuple(relation[:2])
                span1 = tuple(relation[2:4])
                rel = relation[-1]
                key = True
                # key = True
                phrase1 = ' '.join(sentences[span0[0]:(span0[1]+1)])
                phrase2 = ' '.join(sentences[span1[0]:(span1[1]+1)])
                # if phrase1 in phraseset or phrase2 in phraseset:
                #     key = True
                if phrase1.isupper():
                    # print phrase1
                    key = True
                if phrase2.isupper():
                    # print phrase2
                    key = True
                # if phrase1 in set(pronoun) or phrase2 in set(pronoun):
                #     key = True
                if key:
                    relation_token = doc_key + str(span0[0]) + '_' + str(span0[1]) +'_'+ doc_key + str(span1[0]) + '_' + str(span1[1]) +'_' +  rel
                    if relation_token not in trueset:
                        print doc_key
                        print phrase1 +'\t'+ phrase2 +'\t'+ rel
    return true_rels

predfn = '/home/yiluan/lsgn_cleanup/dev.output_nocoref.json'
# predfn = '/home/yiluan/lsgn_cleanup/dev.output.json'
truefn = './ScienceKG_dev.noreverse.json'
true_docs = ReadJson(truefn)
pred_docs = ReadJsonPred(predfn, true_docs)
for key in true_docs:
    true_docs[key]['sentences'] = pred_docs[key]['sentences']
# pdb.set_trace()
# aspects = ["Task", "Generic", "Metric", "Material", "OtherScientificTerm", "Method"]
aspects = ['Generic']
pronoun = ['this', 'it', 'former', 'latter', 'they','those','It','alternative']
# for aspect in aspects:
#     true_rel = GetRel(true_docs, true_docs,aspect)
#     pred_rel = GetRel(pred_docs, true_docs,aspect)
#     print aspect
#     print len(true_rel)
#     print span_metric(true_rel, pred_rel)
# for aspect in aspects:
#     true_rel = GetRelPhrase(true_docs, true_docs,aspect, set(pronoun))
#     pred_rel = GetRelPhrase(pred_docs, true_docs,aspect, set(pronoun))
#     print aspect
#     print len(true_rel)
#     print span_metric(true_rel, pred_rel)
# for aspect in aspects:
#     true_rel = TrueSet(pred_docs, true_docs,aspect)
#     pred_rel = PrintError(true_docs, true_docs,aspect, true_rel)
# for aspect in aspects:
#     true_rel = GetRelCoref(true_docs, true_docs,aspect)
#     pred_rel = GetRelCoref(pred_docs, true_docs,aspect)
#     print aspect
#     print len(true_rel)
#     print span_metric(true_rel, pred_rel)
aspect = 'True'
# true_rel = GetRelSpan(true_docs, true_docs,aspect)
pred_rel = GetRelSpan(pred_docs, true_docs,aspect)

# print span_metric(true_rel, pred_rel) 
