string = '#!/bin/sh'
lst = ['scientific_relation_ffn4','scientific_relation_best_coref','scientific_relation_best_coref1','scientific_relation_best']
print string
for exp in lst:
    print 'python singleton.py ' + exp + ' &'
    print 'python evaluator.py ' + exp + ' > tuning_logs/' + exp + '.log &'
