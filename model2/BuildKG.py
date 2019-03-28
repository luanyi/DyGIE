import json
import pdb
from operator import itemgetter
def ReadJson(senfn, refn, nerfn ,coreffn, docs):
    with open(senfn) as f:
        docs_sent = [json.loads(jsonline) for jsonline in f.readlines()]
    with open(refn) as f:
        docs_re = [json.loads(jsonline) for jsonline in f.readlines()]
    with open(nerfn) as f:
        docs_ner = [json.loads(jsonline) for jsonline in f.readlines()]
    with open(coreffn) as f:
        docs_coref = [json.loads(jsonline) for jsonline in f.readlines()]
    for i in range(len(docs_sent)):
        if docs_sent[i]['doc_key'] != docs_re[i]['doc_key'] or docs_sent[i]['doc_key'] != docs_re[i]['doc_key']:
            pdb.set_trace()
        else:
            year = docs_sent[i]['doc_key'].split('_')[1]
            venue = docs_sent[i]['doc_key'].split('_')[0]
            docs[docs_sent[i]['doc_key']] = {'ner': docs_ner[i]['ner'], 'relation': docs_re[i]['relations'], 'coref':docs_coref[i]['coref'], 'sentences': docs_sent[i]['sentences'], 'year': year, 'venue':venue}
            PropgateHyponym(docs[docs_sent[i]['doc_key']])
            Map2doc(docs[docs_sent[i]['doc_key']])


def ReadJsonACL(senfn, docs):
    with open(senfn) as f:
        docs_sent = [json.loads(jsonline) for jsonline in f.readlines()]

    for i in range(len(docs_sent)):
            year = docs_sent[i]['doc_key'].split('_')[0][1:]
            if year.startswith('0') or year.startswith('1'):
                year = '20' + year
            elif year.startswith('9') or year.startswith('8') or year.startswith('7'):
                year = '19' + year
            
            if len(year) > 4:pdb.set_trace()
            venue = docs_sent[i]['doc_key'].split('_')[0]
            try:
                year =int(year)
            except:
                print docs_sent[i]['doc_key']
                return
            docs[docs_sent[i]['doc_key']] = {'ner': docs_sent[i]['ner'], 'relation': docs_sent[i]['relations'], 'coref':[], 'sentences': docs_sent[i]['sentences'], 'year': year, 'venue':'ACL'}

            Map2doc(docs[docs_sent[i]['doc_key']])

def ReadJsonTitle(senfn, nerfn , docs):
    with open(senfn) as f:
        docs_sent = [json.loads(jsonline) for jsonline in f.readlines()]

    with open(nerfn) as f:
        docs_ner = [json.loads(jsonline) for jsonline in f.readlines()]
    for i in range(len(docs_sent)):
        if docs_sent[i]['doc_key'] != docs_ner[i]['doc_key']:
            pdb.set_trace()
        else:
            year = docs_sent[i]['doc_key'].split('_')[1]
            venue = docs_sent[i]['doc_key'].split('_')[0]
            relations = CreateRelation(docs_ner[i]['ner'])
            docs[docs_sent[i]['doc_key']] = {'ner': docs_ner[i]['ner'],'relation': relations, 'sentences': docs_sent[i]['sentences'], 'coref':[], 'year': year, 'venue':venue}

            # PropgateHyponym(docs[docs_sent[i]['doc_key']])
            Map2doc(docs[docs_sent[i]['doc_key']])

def CreateRelation(sents):
    all_rel = []
    for sent in sents:
        nerdir = {}
        for ner in sent:
            if ner[2] not in nerdir:
                nerdir[ner[2]] = []
            nerdir[ner[2]].append(ner[:2])
        rels = []
        if 'Task' in nerdir and 'Method' in nerdir:
            for span1 in nerdir['Task']:
                for span2 in nerdir['Method']:
                    rel = span2 + span1 + ['USED-FOR']
                    rels.append(rel)
        all_rel.append(rels)
    return all_rel
            
def PropgateHyponym(doc):
    for i, sent in enumerate(doc['relation']):
        hyp_dir = {}
        new_rels = []
        rel_span_sets = set()
        for relation in sent:
            if relation[-1] == 'HYPONYM-OF':

                if tuple(relation[2:4]) in hyp_dir:
                    hyp_dir[tuple(relation[2:4])].append(tuple(relation[:2]))
                else:
                    hyp_dir[tuple(relation[2:4])] = [tuple(relation[:2])]
            rel_span_sets.add(tuple(relation[:4]))
            rel_span_sets.add(tuple(relation[2:4] + relation[:2]))
        if hyp_dir:
            for relation in sent:
                if relation[-1] == 'HYPONYM-OF':continue
                span1 = tuple(relation[:2])
                span2 = tuple(relation[2:4])
                rel = relation[-1]
                if span1 in hyp_dir:
                    for new_span in hyp_dir[span1]:
                        new_rel_span = list(new_span) + list(span2)
                        if tuple(new_rel_span) not in rel_span_sets:
                            new_rels.append(new_rel_span + [rel])
                if span2 in hyp_dir:
                    for new_span in hyp_dir[span2]:
                        new_rel_span = list(span1) + list(new_span)
                        if tuple(new_rel_span) not in rel_span_sets:
                            new_rels.append(new_rel_span + [rel])
            if new_rels:
                doc['relation'][i] += new_rels
def Map2doc(doc):
    flat_sentences = []
    flat_sentences_id = []
    flat_token_id = []
    map_token_id = []
    i = 0
    flat_ners = []
    flat_relations = []
    for idx, sent in enumerate(doc['sentences']):
        flat_sentences += sent
        sentids = []
        for word in sent:
            sentids.append(i)
            i += 1
        map_token_id.append(sentids)
        for ner in doc['ner'][idx]:
            start = sentids[ner[0]]
            end = sentids[ner[1]]
            flat_ners.append([start,end,ner[2]])
        for relation in doc['relation'][idx]:
            start1 = sentids[relation[0]]
            end1 = sentids[relation[1]]
            start2 = sentids[relation[2]]
            end2 = sentids[relation[3]]
            flat_relations.append([start1,end1,start2,end2,relation[-1]])
    doc['sentences'] = flat_sentences
    doc['relation'] = flat_relations
    doc['ner'] = flat_ners
    BuildKG(doc)

def BuildKG(doc):
    NERdir = {}
    RELdir = {}
    for ner in doc['ner']:
        phrase = ' '.join(doc['sentences'][ner[0]:(ner[1]+1)])
        if phrase == 'system' or phrase == 'systems':
            ner[2] = 'Generic'
        
        NERdir[(ner[0],ner[1])] = [ner[2], phrase]
    for relation in doc['relation']:
        start1 = relation[0]
        end1 = relation[1]
        start2 = relation[2]
        end2 = relation[3]
        ntype1 = 'None'
        ntype2 = 'None'
        phrase1 = ' '.join(doc['sentences'][start1:(end1+1)])
        phrase2 = ' '.join(doc['sentences'][start2:(end2+1)])
        if (start1, end1) in NERdir:
            ntype1 = NERdir[(start1, end1)][0]
        else:
            for ner in NERdir:
                if (start1 > ner[0] and start1 < ner[1]) or (end1 > ner[0] and end1 < ner[1]):
                    ntype1 = NERdir[ner][0]+'_partial'
                break
        if (start2, end2) in NERdir:
            ntype2 = NERdir[(start2, end2)][0]
        else:
            for ner in NERdir:
                if (start2 > ner[0] and start2 < ner[1]) or (end2 > ner[0] and end2 < ner[1]):
                    ntype2 = NERdir[ner][0]+'_partial'
                break
        RELdir[(start1, end1,start2, end2)] = [relation[-1],(ntype1,ntype2),(phrase1, phrase2)]
    

            

    doc['NERdir'] = NERdir
    doc['RELdir'] = RELdir

    
def sort_dict(dictionary):
    sorted_dct = sorted(dictionary.items(), key=itemgetter(1),reverse=True)
    return sorted_dct


def ReadTopLst(venue_type, topnum):
    top_dir = {}
    top_dir_len = {}
    top_dir_count = {}
    types = ['Method','Task']

    for ntype in types:
        fn = './NER_analy/' + venue_type + '_' + ntype + '.rank'
        i = 0
        for line in open(fn):
            if i > topnum:break
            phrase, count = line.rstrip().split('\t')
            newphrase = phrase.replace('-','').replace(' ','')
            i += 1
            if newphrase not in top_dir_count or count > top_dir_count[newphrase][0]:
                top_dir_len[phrase] = len(newphrase)
                top_dir[phrase] = newphrase
                top_dir_count[newphrase] = [count, phrase]
            else:
                continue
    top_dir_len = sort_dict(top_dir_len)
    toplst = []
    for token in top_dir_len:
        phrase = token[0]
        toplst.append([top_dir[phrase], phrase])
    return toplst
        
def ReadTopLsts(venue_types, topnum, acronym_dir):
    top_dir = {}
    top_dir_len = {}
    top_dir_count = {}
    types = ['Method','Task']
    for venue_type in venue_types:
        for ntype in types:
            fn = './NER_analy/' + venue_type + '_' + ntype + '.rank'
            i = 0
            for line in open(fn):
                if i > topnum:break
                phrase, count = line.rstrip().split('\t')
                words = []
                for word in phrase.split():
                    if word.isupper() and word in acronym_dir and len(phrase.split()) > 1:
                        words.append(acronym_dir[word])
                    else:
                        words.append(word)
                phrase = ' '.join(words)
                newphrase = phrase.replace('-','').replace(' ','')
                i += 1
                if newphrase not in top_dir_count or count > top_dir_count[newphrase][0]:
                    top_dir_len[phrase] = len(newphrase)
                    top_dir[phrase] = newphrase
                    top_dir_count[newphrase] = [count, phrase]
                else:
                    continue
    top_dir_len = sort_dict(top_dir_len)
    toplst = []
    for token in top_dir_len:
        phrase = token[0]
        toplst.append([top_dir[phrase], phrase])
    return toplst
        
        
        
            
def NormalizedLst(docs, aspects='None', aspect_values = 'None'):
    # aspects = ['year','venue'], aspect_values = [set('1988','2000'),set('EMNLP','ACL')]
    ner_rankdir = {}
    venue_set = set()
    for doc_key in docs:
        doc = docs[doc_key]
        key = False
        if aspects != 'None':
            for i in range(len(aspects)):
                aspect = aspects[i]
                aspect_value = aspect_values[i]
                if doc[aspect] not in aspect_value:
                    key = True
                    break
        if key:continue
        venue_set.add(doc['year'] + '_' + doc['venue'])
        for span in doc['NERdir']:
            phrase = doc['NERdir'][span]
            if phrase[0] == 'Generic':continue
            if phrase[1].endswith('system') or phrase[1].endswith('systems'):
                phrase[0] = 'Task'
            if phrase[0] not in ner_rankdir:
                ner_rankdir[phrase[0]] = {}
            if phrase[1] not in ner_rankdir[phrase[0]]:
                ner_rankdir[phrase[0]][phrase[1]] = 0
            ner_rankdir[phrase[0]][phrase[1]] += 1
    ner_rankdir = CombineDir(ner_rankdir)
    acronym_dir = GetAcronymDirNoType(ner_rankdir)
    rankdir = FilterNERNotype(ner_rankdir,acronym_dir)
    return rankdir, acronym_dir
    # for ner_type in ner_rankdir:
    #     ner_rankdir[ner_type] = sort_dict(ner_rankdir[ner_type])


def NormalizePhrase(phrase, rankdir, acronym_dir, toplst):
    replacewords = set(['model','approach','method','algorithm','technique','module', 'application','models','approachs','methods','algorithms','techniques','modules', 'applications', 'problem','problems','task','tasks', 'system', 'systems', 'score', 'scores','framework','frameworks', 'design', 'designs', 'formulation'])
    words = []
    phrase = phrase.split()
    for word in phrase:
        if word in acronym_dir and word not in set(['-LRB-','-RRB-']):
            word = acronym_dir[word]
        words.append(word)
    phrase = ' '.join(words)
    if '-LRB-' in phrase and '-RRB-' in phrase:
    # if '-LRB-' in phrase and phrase.endswith('-RRB-'):
        full = phrase.split('-LRB-')[0]
        full = Lower(full)
        
        if full in acronym_dir:
            full = acronym_dir[full]
        fullnorm = full.replace('-','').replace(' ','')
        for norm in toplst:
            if norm[0] in fullnorm:
                return norm[1]
        lastword = full.split()[-1]
        if lastword in replacewords:
            full = ' '.join(full.split()[:-1])
            if full in acronym_dir:
                full = acronym_dir[full]

                
        if full in rankdir:
            return full
        if full.endswith('s'):
            if full[:-1] in rankdir:
                return full[:-1]
        if full.endswith('es'):
            if full[:-2] in rankdir:
                return full[:-2]
        return full
    else:
        full = Lower(phrase)
        if full in acronym_dir:
            full = acronym_dir[full]
        fullnorm = full.replace('-','').replace(' ','')
        for norm in toplst:
            if norm[0] in fullnorm:
                return norm[1]
                                            
        lastword = full.split()[-1]
        # if lastword in replacewords:
        #     newfull = ' '.join(full.split()[:-1])
        #     if newfull in rankdir:
        #         return newfull
        if lastword in replacewords:
            full = ' '.join(full.split()[:-1])

        if full.endswith('s'):
            if full[:-1] in rankdir:
                
                return full[:-1]
        if full.endswith('es'):
            if full[:-2] in rankdir:
                return full[:-2]
        if full in rankdir:
            return full
        return full
        # acronym = Lower(acronym)
    
    
    
def topNER(docs, aspects='None', aspect_values = 'None'):
    # aspects = ['year','venue'], aspect_values = [set('1988','2000'),set('EMNLP','ACL')]
    ner_rankdir = {}
    venue_set = set()
    for doc_key in docs:
        doc = docs[doc_key]
        key = False
        if aspects != 'None':
            for i in range(len(aspects)):
                aspect = aspects[i]
                aspect_value = aspect_values[i]
                if doc[aspect] not in aspect_value:
                    key = True
                    break
        if key:continue
        venue_set.add(doc['year'] + '_' + doc['venue'])
        for span in doc['NERdir']:
            phrase = doc['NERdir'][span]
            if phrase[0] == 'Generic':continue
            phrase[1] = Lower(phrase[1])
            if phrase[1].endswith('system') or phrase[1].endswith('systems'):
                phrase[0] = 'Task'
            words = phrase[1].split(' ')
            if phrase[0] not in ner_rankdir:
                ner_rankdir[phrase[0]] = {}
            if phrase[1] not in ner_rankdir[phrase[0]]:
                ner_rankdir[phrase[0]][phrase[1]] = 0
            ner_rankdir[phrase[0]][phrase[1]] += 1
    acronym_dir = GetAcronymDir(ner_rankdir)
    ner_rankdir = FilterNER(ner_rankdir,acronym_dir)

    for ner_type in ner_rankdir:
        ner_rankdir[ner_type] = sort_dict(ner_rankdir[ner_type])

    print venue_set
    return ner_rankdir

def CombineDir(dct):
    new_dct = {}
    for aspect in dct:
        for phrase in dct[aspect]:
            if phrase in new_dct:
                new_dct[phrase] += dct[aspect][phrase]
            else:
                new_dct[phrase] = dct[aspect][phrase]
    return new_dct
def GetAcronymDir(ner_rankdir):
    acronym_counts = {}
    for ner_type in ner_rankdir:
        for key in ner_rankdir[ner_type]:
            # if '-LRB-' in key and key.endswith('-RRB-'):
            if '-LRB-' in key and '-RRB-' in key:
                full = key.split('-LRB-')[0]
                acronym = key.split('-LRB-')[1].split('-RRB-')[0]
                if len(full) < len(acronym):
                    full,acronym = acronym,full
                full = Lower(full)
                acronym = Lower(acronym)
                if acronym not in acronym_counts:
                    acronym_counts[acronym] = {full:1}
                else:
                    if full in acronym_counts[acronym]:
                        acronym_counts[acronym][full] += 1
                    else:
                        acronym_counts[acronym][full] = 1
    new_acronym_counts = {}
    for acronym in acronym_counts:
            # acronym_counts[acronym] = sort_dict(acronym_counts[acronym])[0][0]
        acronym_counts[acronym] = MergePlural(acronym_counts[acronym])
        sorted_result = sort_dict(acronym_counts[acronym])[0]
        if sorted_result[1] < 2:
            continue
        else:
            new_acronym_counts[acronym] = sorted_result[0]

    return new_acronym_counts

def GetAcronymDirNoType(ner_rankdir):
    acronym_counts = {}

    for key in ner_rankdir:
            if '-LRB-' in key and '-RRB-' in key:
                full = key.split('-LRB-')[0]
                acronym = key.split('-LRB-')[1].split('-RRB-')[0]
                if len(full) < len(acronym):
                    full,acronym = acronym,full
                full = Lower(full)
                acronym = Lower(acronym)
                if acronym not in acronym_counts:
                    acronym_counts[acronym] = {full:1}
                else:
                    if full in acronym_counts[acronym]:
                        acronym_counts[acronym][full] += 1
                    else:
                        acronym_counts[acronym][full] = 1
    new_acronym_counts = {}
    for acronym in acronym_counts:
            # acronym_counts[acronym] = sort_dict(acronym_counts[acronym])[0][0]
        acronym_counts[acronym] = MergePlural(acronym_counts[acronym])
        sorted_result = sort_dict(acronym_counts[acronym])[0]
        if sorted_result[1] < 2:
            continue
        else:
            new_acronym_counts[acronym] = sorted_result[0]

    return new_acronym_counts
                
def MergePlural(mix_dict):
    merged_dict = {}
    plural_mappings = {}
    for key in mix_dict:
        if key + 's' in mix_dict:
            plural_mappings[key+'s'] = key
            continue
        if key + 'es' in mix_dict:
            plural_mappings[key+'es'] = key
            continue
    for key in mix_dict:
        if key in plural_mappings:
            new_key = plural_mappings[key]
            if new_key in merged_dict:
                merged_dict[new_key] += mix_dict[key]
            else:
                merged_dict[new_key] = mix_dict[key]
        else:
            new_key = key
            if new_key in merged_dict:
                merged_dict[new_key] += mix_dict[key]
            else:
                merged_dict[new_key] = mix_dict[key]
            
    return merged_dict


def FilterNER(ner_rankdir, acronym_dir):
    mappings_all = {}
    acro_mappings = {}
    plural_mappings = {}
    for ner_type in ner_rankdir:
        for key1 in ner_rankdir[ner_type]:
            # if '-LRB-' in key1 and key1.endswith('-RRB-'):
            if '-LRB-' in key1 and '-RRB-' in key1:
                full = key1.split('-LRB-')[0]
                acronym = key1.split('-LRB-')[1].split('-RRB-')[0]
                mappings_all[key1] = Lower(full)
                if len(full) < len(acronym):
                    full,acronym = acronym,full
                acro_mappings[full] = acronym
    new_ner_rankdir = {}
    # replace all bracket phrases with their full name
    for ner_type in ner_rankdir:
        new_ner_rankdir[ner_type] = {}
        for key1 in ner_rankdir[ner_type]:
            if key1 in mappings_all:
                if mappings_all[key1] in new_ner_rankdir[ner_type]:
                    new_ner_rankdir[ner_type][mappings_all[key1]] += ner_rankdir[ner_type][key1]
                else:
                    if key1 in ner_rankdir[ner_type]:
                        new_ner_rankdir[ner_type][mappings_all[key1]] = ner_rankdir[ner_type][key1]
                    
            else:
                if key1 in new_ner_rankdir[ner_type]:
                    new_ner_rankdir[ner_type][key1] += ner_rankdir[ner_type][key1]
                else:
                    if key1 in ner_rankdir[ner_type]:
                        new_ner_rankdir[ner_type][key1] = ner_rankdir[ner_type][key1]
                    

    for ner_type in new_ner_rankdir:
        for key1 in new_ner_rankdir[ner_type]:
            if key1 + 's' in new_ner_rankdir[ner_type]:
                plural_mappings[key1+'s'] = key1
                continue
            if key1 + 'es' in new_ner_rankdir[ner_type]:
                plural_mappings[key1+'es'] = key1
                continue


    ner_rankdir_final = {}
    for ner_type in new_ner_rankdir:
        ner_rankdir_final[ner_type] = {}
        for key in new_ner_rankdir[ner_type]:
            new_key = Lower(key)
            if new_key in plural_mappings:
                single_term = plural_mappings[new_key]
                single_term = single_term
                if single_term in acronym_dir:
                    single_term = acronym_dir[single_term]
                if single_term in ner_rankdir_final[ner_type]:
                        ner_rankdir_final[ner_type][single_term] += new_ner_rankdir[ner_type][key]
                        
                else:
                        ner_rankdir_final[ner_type][single_term] = new_ner_rankdir[ner_type][key]

            else:
                lower_key = Lower(key)
                if lower_key in acronym_dir:
                    lower_key = acronym_dir[lower_key]
                    if lower_key in plural_mappings:
                        lower_key = plural_mappings[lower_key]
                    
                if lower_key in ner_rankdir_final[ner_type]:
                    ner_rankdir_final[ner_type][lower_key] += new_ner_rankdir[ner_type][key]
                else:
                    ner_rankdir_final[ner_type][lower_key] = new_ner_rankdir[ner_type][key]
                    
    # pdb.set_trace()
    rankdir = {}
    replacewords = set(['model','approach','method','algorithm','technique','module', 'application','models','approachs','methods','algorithms','techniques','modules', 'applications', 'problem','problems','task','tasks', 'system', 'systems', 'score', 'scores', 'framework','frameworks','design', 'designs'])
    for ner_type in ner_rankdir_final:
        rankdir[ner_type] = {}
        for key in ner_rankdir_final[ner_type]:
            if not key:continue
            words = key.split()
            if words[-1] in replacewords:
                if len(words) == 1:continue 
                new_phrase = ' '.join(words[:-1])
                if new_phrase in acronym_dir:
                    new_phrase = acronym_dir[new_phrase]
                # if new_phrase not in rankdir[ner_type]:
                #     rankdir[ner_type][new_phrase] = 0
                # rankdir[ner_type][new_phrase] += ner_rankdir_final[ner_type][key]
                if new_phrase in ner_rankdir_final[ner_type]:
                    if new_phrase not in rankdir[ner_type]:
                        rankdir[ner_type][new_phrase] = 0
                    rankdir[ner_type][new_phrase] += ner_rankdir_final[ner_type][key]
                else:
                    if key not in rankdir[ner_type]:
                        rankdir[ner_type][key] = 0
                    rankdir[ner_type][key] += ner_rankdir_final[ner_type][key]
                        
            else:
                if key not in rankdir[ner_type]:
                    rankdir[ner_type][key] = 0
                rankdir[ner_type][key] += ner_rankdir_final[ner_type][key]

    return rankdir

def FilterNERNotype(ner_rankdir, acronym_dir):
    mappings_all = {}
    acro_mappings = {}
    plural_mappings = {}
    for key1 in ner_rankdir:
            if '-LRB-' in key1 and '-RRB-' in key1:
                full = key1.split('-LRB-')[0]
                acronym = key1.split('-LRB-')[1].split('-RRB-')[0]
                
                mappings_all[key1] = Lower(full)
                
    new_ner_rankdir = {}
    # replace all bracket phrases with their full name

    for key1 in ner_rankdir:
            if key1 in mappings_all:
                if mappings_all[key1] in new_ner_rankdir:
                    new_ner_rankdir[mappings_all[key1]] += ner_rankdir[key1]
                else:
                    if key1 in ner_rankdir:
                        new_ner_rankdir[mappings_all[key1]] = ner_rankdir[key1]
                    
            else:
                if key1 in new_ner_rankdir:
                    new_ner_rankdir[key1] += ner_rankdir[key1]
                else:
                    if key1 in ner_rankdir:
                        new_ner_rankdir[key1] = ner_rankdir[key1]
                    


    for key1 in new_ner_rankdir:
            if key1 + 's' in new_ner_rankdir:
                plural_mappings[key1+'s'] = key1
            if key1 + 'es' in new_ner_rankdir:
                plural_mappings[key1+'es'] = key1



    ner_rankdir_final = {}
    for key in new_ner_rankdir:
            new_key = Lower(key)
            if new_key in plural_mappings:
                single_term = plural_mappings[new_key]
                single_term = Lower(single_term)
                if single_term in acronym_dir:
                    single_term = acronym_dir[single_term]
                if single_term in ner_rankdir_final:
                        ner_rankdir_final[single_term] += new_ner_rankdir[key]
                        
                else:
                        ner_rankdir_final[single_term] = new_ner_rankdir[key]
            else:
                lower_key = new_key
                if lower_key in acronym_dir:
                    lower_key = acronym_dir[lower_key]
                    if lower_key in plural_mappings:
                        lower_key = plural_mappings[lower_key]
                if lower_key in ner_rankdir_final:
                    ner_rankdir_final[lower_key] += new_ner_rankdir[key]
                else:
                    ner_rankdir_final[lower_key] = new_ner_rankdir[key]
                    

    return ner_rankdir_final

def Lower(string):
    words = string.split()
    new_string = []
    for word in words:
        if not word:continue
        # if not word.isupper():
        #     new_string.append(word.lower())
        # else:
        #     new_string.append(word)
        if word[:1].isupper() and word[1:].islower():
            new_string.append(word.lower())
        else:
            new_string.append(word)
    return ' '.join(new_string)
            
def CountMissingNER(docs):
    allnum = 0
    overlapnum = 0
    nonnum = 0
    ner_labels = ["Task", "Generic", "Metric", "Material", "OtherScientificTerm", "Method"]
    for doc in docs:
        for rel in docs[doc]['RELdir']:
            rel = docs[doc]['RELdir'][rel]
            if 'None' == rel[1][0]:
                nonnum += 1
                # print rel[2][0]
            elif 'partial' in rel[1][0]:
                overlapnum += 1
            elif rel[1][0] not in ner_labels:
                print rel[1][0]
            if 'None' in rel[1][1]:
                nonnum += 1
            elif 'partial' in rel[1][1]:
                overlapnum += 1
            elif rel[1][1] not in ner_labels:
                print rel[1][1]
            # if rel[1][0] in ner_labels:
            #     if rel[1][0] == 'Task':
            #         print rel[2][0]
            allnum += 2
    print allnum
    print overlapnum
    print nonnum
    print float(nonnum)/allnum
    print float(overlapnum)/allnum

def PrintK(dictionary, aspect, k , name):
    strings = []
    k = min(k,len(dictionary[aspect]))
    for i in range(k):
        strings.append(dictionary[aspect][i][0] + '\t' + str(dictionary[aspect][i][1]))
    fid = open('./NER_analy/' + name + '_'+ aspect + '.rank','w')
    fid.write('\n'.join(strings).encode('utf-8'))
    fid.close()
    

def VoteRelationType(rel_dir_phrase):
    phrase_rel_dir = {}
    phrase_ner_dir = {}
    phrase_count = {}
    for rel in rel_dir_phrase:
        for aspect in rel_dir_phrase[rel]:

            for phrase in rel_dir_phrase[rel][aspect]:
                if phrase in phrase_count:
                    phrase_count[phrase] += rel_dir_phrase[rel][aspect][phrase]
                else:
                    phrase_count[phrase] = rel_dir_phrase[rel][aspect][phrase]
                if phrase not in phrase_rel_dir:
                    phrase_rel_dir[phrase] = {}
                if rel not in phrase_rel_dir[phrase]:
                    phrase_rel_dir[phrase][rel] = 0
                if phrase not in phrase_ner_dir:
                    phrase_ner_dir[phrase] = {}
                if aspect not in phrase_ner_dir[phrase]:
                    phrase_ner_dir[phrase][aspect] = 0
                phrase_rel_dir[phrase][rel] += rel_dir_phrase[rel][aspect][phrase]
                phrase_ner_dir[phrase][aspect] += rel_dir_phrase[rel][aspect][phrase]
                    

    new_dict = {}
    for phrase in phrase_count:
        phrase_rel_dir[phrase] = sort_dict(phrase_rel_dir[phrase])
        phrase_ner_dir[phrase] = sort_dict(phrase_ner_dir[phrase])
        rel = phrase_rel_dir[phrase][0][0]
        aspect = phrase_ner_dir[phrase][0][0]
        if len(phrase_rel_dir[phrase])> 1:
            if rel == 'CONJUNCTION':
                rel = phrase_rel_dir[phrase][1][0]
        if len(phrase_ner_dir[phrase])> 1:
            if aspect == 'None' or aspect == 'OtherScientificTerm':
                aspect = phrase_ner_dir[phrase][1][0]
            if aspect == 'None' or aspect == 'OtherScientificTerm':
                if len(phrase_ner_dir[phrase])> 2:
                    aspect = phrase_ner_dir[phrase][2][0]
                
        # for token in phrase_dir[phrase]:
        #     if ('USED-FOR', 'Task') == token[0]:
        #         (rel,aspect) = ('USED-FOR', 'Task')
        #     elif ('USED-FOR_Reverse', 'Method') == token[0]:
        #         (rel,aspect) = ('USED-FOR_Reverse', 'Method')
                

        if rel not in new_dict:
            new_dict[rel] = {}
        if aspect not in new_dict[rel]:
            new_dict[rel][aspect] = {}
        new_dict[rel][aspect][phrase] = phrase_count[phrase]

    return new_dict
        
    
    
    
# docs = {}
# for i in range(16):
#     senfn = '/homes/luanyi/pubanal/data/AI2/json/'+str(i)+'.json'
#     nerfn = '/homes/luanyi/pubanal/data/AI2/automatic_graph/results/ner_outputs/'+str(i)+'.output.json'
#     refn = '/homes/luanyi/pubanal/data/AI2/automatic_graph/results/re_outputs/'+str(i)+'.output.json'
#     ReadJson(senfn, refn, nerfn, docs)
    
# venue_sets = [set(['ICASSP','INTERSPEECH']), set(['AAAI','IJCAI']), set(['ACL','EMNLP','IJCNLP']), set(['ECCV','CVPR','ICCV']), set(['NIPS','ICML'])]
# speech = set(['ICASSP','INTERSPEECH'])
# AI = set(['AAAI','IJCAI'])
# NLP = set(['ACL','EMNLP','IJCNLP'])
# CV = set(['ECCV','CVPR','ICCV'])
# ML = set(['NIPS','ICML'])
# # speech = topNER(docs, ['venue'], [venue_sets[0]])
# # AI = topNER(docs, ['venue'], [venue_sets[1]])
# # NLP = topNER(docs, ['venue'], [venue_sets[2]])
# # CV = topNER(docs, ['venue'], [venue_sets[3]])
# # ML = topNER(docs, ['venue'], [venue_sets[4]])
# b00 = set([str(i) for i in range(1990,2001)])
# p00_to_05 = set([str(i) for i in range(2001,2006)])
# p06_to_10 = set([str(i) for i in range(2006,2011)])
# p10_to_17 = set([str(i) for i in range(2011,2017)])
# # p1 = topNER(docs, ['year'], [b00])
# # p2 = topNER(docs, ['year'], [p00_to_05])
# # p3 = topNER(docs, ['year'], [p06_to_10])
# # p4 = topNER(docs, ['year'], [p10_to_17])
# # p1_speech = topNER(docs, ['year','venue'], [b00,venue_sets[0]])
# # p2_speech = topNER(docs, ['year','venue'], [p00_to_05, venue_sets[0]])
# # p3_speech = topNER(docs, ['year','venue'], [p06_to_10, venue_sets[0]])
# p4_speech = topNER(docs, ['year','venue'], [p10_to_17, venue_sets[0]])
# # p1_ai = topNER(docs, ['year','venue'], [b00,venue_sets[1]])
# # p2_ai = topNER(docs, ['year','venue'], [p00_to_05, venue_sets[1]])
# # p3_ai = topNER(docs, ['year','venue'], [p06_to_10, venue_sets[1]])
# # p4_ai = topNER(docs, ['year','venue'], [p10_to_17, venue_sets[1]])
# # p1_NLP = topNER(docs, ['year','venue'], [b00,venue_sets[2]])
# # p2_NLP = topNER(docs, ['year','venue'], [p00_to_05, venue_sets[2]])
# # p3_NLP = topNER(docs, ['year','venue'], [p06_to_10, venue_sets[2]])
# # p4_NLP = topNER(docs, ['year','venue'], [p10_to_17, venue_sets[2]])
# # p1_cv = topNER(docs, ['year','venue'], [b00,venue_sets[3]])
# # p2_cv = topNER(docs, ['year','venue'], [p00_to_05, venue_sets[3]])
# # p3_cv = topNER(docs, ['year','venue'], [p06_to_10, venue_sets[3]])
# # p4_cv = topNER(docs, ['year','venue'], [p10_to_17, venue_sets[3]])
# # p1_ml = topNER(docs, ['year','venue'], [b00,venue_sets[4]])
# # p2_ml = topNER(docs, ['year','venue'], [p00_to_05, venue_sets[4]])
# # p3_ml = topNER(docs, ['year','venue'], [p06_to_10, venue_sets[4]])
# # p4_ml = topNER(docs, ['year','venue'], [p10_to_17, venue_sets[4]])
# years = {'p1':b00, 'p2':p00_to_05, 'p3':p06_to_10, 'p4':p10_to_17}
# venues = {'speech':speech, 'AI':AI, 'NLP':NLP, 'CV':CV, 'ML':ML}
# ner_types = ["Task", "Generic", "Metric", "Material", "OtherScientificTerm", "Method"]
# # rankdir, acronym_dir = NormalizedLst(docs, ['year','venue'], [years['p4'], venues['speech']])
# # phrase = 'Gaussian mixture models'
# # NormalizePhrase(phrase, rankdir, acronym_dir)

# # pdb.set_trace()
# print 'aa'
# # topNER(docs, ['year','venue'], [years['p4'], venues['speech']])
# print 'finish'
# for year_key in years:
#     year = years[year_key]
#     print year
#     for venue_key in venues:
#         venue = venues[venue_key]
#         x = topNER(docs, ['year','venue'], [year, venue])
#         for ner_type in ner_types:
#             PrintK(x,ner_type,1000, venue_key + '_' + year_key)
        
# for year_key in years:
#     print year_key
#     year = years[year_key]
#     x = topNER(docs, ['year'], [year])
#     for ner_type in ner_types:
#         PrintK(x,ner_type,1000, year_key)
# for venue_key in venues:
#     venue = venues[venue_key]
#     x = topNER(docs, ['venue'], [venue])
#     for ner_type in ner_types:
#         PrintK(x,ner_type,1000, venue_key)
# # CountMissingNER(docs)
