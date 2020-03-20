transitions = {}
emmisions = {}
observations = [[]]
states = ('(0,0)','(0,1)','(1,0)','(1,1)')
def main():
    with open('q4hmm.txt') as f:
        i = 0
        data = f.readlines()
        for line in data:
            i += 1
            line = line.rstrip().split(' ')
            if i > 2 and i <= 22:
                # load Transition
                if (line[0] not in transitions.keys()):
                    transitions[line[0]] = {}
                transitions[line[0]].update({line[1]: float(line[2])})
            elif i >= 25:
                # load Emmision
                if (line[0] not in emmisions.keys()):
                    emmisions[line[0]] = {}
                emmisions[line[0]].update({line[1]: float(line[2])})

    with open('q4.dat') as f:
        data = f.readlines()
        for line in data:
            line = line.rstrip()
            if line == '':
                observations.append([])
            else:
                observations[-1].append(line)
        observations.pop()

    print(observations)

    print(" FORWARD PROBABILITIES ")
    for i, obsrv in enumerate(observations):
        print_prob(i,len(obsrv),forward(obsrv))
    
    print(" BACKWARD PROBABILITIES ")
    for i, obsrv in enumerate(observations):
        print_prob(i,len(obsrv), backward(obsrv))

def forward(observ):
    forward_prob = []
    h_prev = {}
    for i, observ_i in enumerate(observ):
        h_c = {}
        for state in states:
            if i == 0:
                total_h = transitions['(None,None)'][state]
            else:
                total_h = sum(h_prev[j] * transitions[j][state] for j in states)
            h_c[state] = emmisions[state][observ_i] * total_h
        forward_prob.append(h_c)
        h_prev = h_c
    return forward_prob

def backward(obsrv):
    backward_prob = []
    h_prev = {}
    for i, observ_i in enumerate(reversed(obsrv)):
        h_c = {}
        for state in states:
            if i == 0:
                h_c[state] = 1
            else:
                h_c[state] = sum(transitions[state][j] * emmisions[j][observ_i] * h_prev[j] for j in states)
        backward_prob.insert(0, h_c)
        h_prev = h_c
    return backward_prob

def print_prob(n_seq, length, probabilities):
    print("Sequence {0}, length {1}".format(n_seq,length))
    for state in states:
        line = state
        for prob in probabilities: 
            line += ' {:.3f}'.format(prob[state])
        print(line)
    print()

if __name__ == '__main__':
    main()
