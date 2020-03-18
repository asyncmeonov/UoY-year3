#!/usr/bin/env python

import sys
from copy import deepcopy
from math import log


fobj = open(sys.argv[1])
vs = fobj.readline().rstrip().split()
arities = [int(x) for x in fobj.readline().rstrip().split()]
data = []
for line in fobj:
    # make all values integers then can use lists rather than
    # dictionaries
    data.append([int(x) for x in line.rstrip().split()])
n = len(data)
    
for i, thisv in enumerate(vs):
    thisX = [dpt[:i]+dpt[i+1:] for dpt in data]
    thisY = [dpt[i] for dpt in data]

    # initialise for class frequencies
    class_counts = [0]*arities[i]

    # make a 3D list of zeroes
    # counts[y][k][v] is how often feature k took value v
    # when the class was y
    # y,k and v are all non-negative integers so we can use
    # them as list indices
    tmp = []
    for j in range(i):
        tmp.append([0]*arities[j])
    for j in range(i+1,len(vs)):
        tmp.append([0]*arities[j])
    counts = []
    for j in range(arities[i]):
        counts.append(deepcopy(tmp))

    # get counts from data
    for j, y in enumerate(thisY):
        class_counts[y] += 1
        for k, v in enumerate(thisX[j]):
            counts[y][k][v] += 1

    # replace counts with corresponding log probability
    # ( could just overwrite 'counts' but creating separate 'logprob' data structure is clearer )
    # use the value None to stand for -infinity
    # use the name 'klass' to avoid confusion with 'class' reserved word
    logprob = deepcopy(counts)
    classlogprob = []
    majority_class, majority_class_count = None, 0
    for klass, klass_count in enumerate(class_counts):
        if klass_count == 0:
            # never saw this class in the data
            classlogprob.append(None)
            continue
        classlogprob.append(log(klass_count)/n)
        if klass_count > majority_class_count:
            majority_class, majority_class_count = klass, klass_count
        for feature, featurevaluecounts in enumerate(counts[klass]):
            for featureval, count in enumerate(featurevaluecounts):
                logprob[klass][feature][featureval] = log(count/klass_count) if count > 0 else None

    # now start predicting ....
    nerrs = 0
    majority_class_nerrs = 0
    for i, testpt in enumerate(thisX):
        maxlprob, mostprobclass = None, None
        # consider each possible class
        for klass, clslogprob in enumerate(classlogprob):
            if clslogprob is None:
                # this class has zero probability
                continue

            # initialise sum of log probabilities to class log probability 
            lp = clslogprob
            # consider each feature and add log probability depending on its value
            # (and current class)
            # if any log probabilities are -infinity (represented by None)
            # the current class has probability zero, so at (*) set log probability to None and break
            for feature, featureval in enumerate(testpt):
                thislp = logprob[klass][feature][featureval] 
                if thislp is None:
                    # (*)
                    lp = None
                    break
                else:
                    lp += thislp
            if lp is not None and (mostprobclass is None or lp > maxlprob):
                # update most probable class
                # NB. maxlprob is actually log(P(class,f1,...fp)) not log(P(class|f1,...fp))
                # the former is all we need to work out which class is most probable
                maxlprob, mostprobclass = lp, klass

        # should not have zero probability for all classes even though using MLEs
        # since testing on the training data!
        assert mostprobclass is not None
        
        if mostprobclass != thisY[i]:
            nerrs += 1
        if majority_class != thisY[i]:
            majority_class_nerrs += 1

    print('{0}: naive Bayes errors = {1}, majority class errors = {2}'.format(
        thisv,nerrs,majority_class_nerrs))

