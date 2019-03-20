from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np

#Script to evalutate relation
#input: grelations:gold relation list
#       prelations:predicted relation list
#       grelations: [rel1, rel2..]
#       rel1 = [[entity1_span, entity2_span], relation_type]
#       entity1_span = 'startid_endid' #should be string
#       e.g. [['1_2', '4_8'], 'COMPARE']

def span_metric(grelations, prelations):
    g_spans = []
    p_spans = []
    res_gold = []
    res_pred = []
    for rel in grelations:
        if 'REVERSE' in rel[1]:
            span = rel[0][1] + '_' + rel[0][0]
            g_spans.append(span)
        else:
            g_spans.append('_'.join(rel[0]))
        res_gold.append(rel[1])
            
    for rel in prelations:
        if 'REVERSE' in rel[1]:
            span = rel[0][1] + '_' + rel[0][0]
            p_spans.append(span)
        else:
            p_spans.append('_'.join(rel[0]))
        res_pred.append(rel[1])
            
    spans_all = set(p_spans + g_spans)
    res_all_gold = []
    res_all_pred = []
    targets = set()
    for i, r in enumerate(spans_all):
        if r in g_spans:
            target = res_gold[g_spans.index(r)].split("_")[0]
            res_all_gold.append(target)
            targets.add(target)
        else:
            res_all_gold.append('None')
        if r in p_spans:
            target = res_pred[p_spans.index(r)].split("_")[0]
            res_all_pred.append(target)
            targets.add(target)
        else:
            res_all_pred.append('None')
    targets = list(targets)
    prec, recall, f1, support = precision_recall_fscore_support(res_all_gold, res_all_pred, labels=targets, average=None)
    metrics = {}
    metrics = {}
    for k, target in enumerate(targets):
        metrics[target] = {
            'precision': prec[k],
            'recall': recall[k],
            'f1-score': f1[k],
            'support': support[k]
        }
    prec, recall, f1, s = precision_recall_fscore_support(res_all_gold, res_all_pred, labels=targets, average='micro')
    
    metrics['overall'] = {
        'precision': prec,
        'recall': recall,
        'f1-score': f1,
        'support': sum(support)
    }
    print_report(metrics, targets)        
    return prec, recall, f1

def print_report(metrics, targets, digits=2):
    def _get_line(results, target, columns):
        line = [target]
        for column in columns[:-1]:
            line.append("{0:0.{1}f}".format(results[column], digits))
        line.append("%s" % results[columns[-1]])
        return line

    columns = ['precision', 'recall', 'f1-score', 'support']

    fmt = '%11s' + '%9s' * 4 + '\n'
    report = [fmt % tuple([''] + columns)]
    report.append('\n')
    for target in targets:
        results = metrics[target]
        line = _get_line(results, target, columns)
        report.append(fmt % tuple(line))
    report.append('\n')

    # overall
    line = _get_line(metrics['overall'], 'avg / total', columns)
    report.append(fmt % tuple(line))
    report.append('\n')

    print(''.join(report))

