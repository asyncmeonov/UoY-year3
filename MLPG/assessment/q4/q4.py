# transitions = {"From":[], "To":[], "Prob":[]}
# emmisions = {"State":[], "Symbol":[], "Prob":[]}
transitions = {}
emmisions = {}
observations = [[]]
# states = ('(0,0)','(0,1)','(0,2)','(0,3)','(1,0)','(1,1)','(1,2)','(1,3)','(2,1)','(2,3)','(3,0)','(3,1)','(3,2)','(3,3)')
states = ('(0,0)','(0,1)','(1,0)','(1,1)')
def main():
    with open('q4hmm.txt') as f:
        i = 0
        data = f.readlines()
        for line in data:
            i += 1
            line = line.rstrip().split(' ')
            if i > 2 and i <= 22:
                # Do Transition
                if (line[0] not in transitions.keys()):
                    transitions[line[0]] = {}
                transitions[line[0]].update({line[1]: float(line[2])})
            elif i >= 25:
                # Do Emmision
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

    print(transitions)
    print("="*20)
    print("-"*20)
    print("="*20)
    print(emmisions)
    #print prob for a given state transition
    print(sum(transitions['(None,None)'].values()))

    # test
    # test = observations[0]
    # print(test)
    # print(test[1:])
    # print(reversed(test[1:]))
    # print(reversed(test[1:]+(None,)))

    print(" FORWARD PROBABILITIES ")
    print_prob(0,len(observations[0]),forward(observations[0]))
    print(" BACKWARD PROBABILITIES ")
    print_prob(0,len(observations[0]), backward(observations[0]))

def forward(observ):
    forward_prob = []
    h_prev = {}
    for i, observ_i in enumerate(observ):
        h_c = {}
        for state in states:
            if i == 0:
                print("Starting state...")
                total_h = transitions['(None,None)'][state]
            else:
                total_h = sum(h_prev[j] * transitions[j][state] for j in states)
            h_c[state] = emmisions[state][observ_i] * total_h
        forward_prob.append(h_c)
        h_prev = h_c
    return forward_prob
# uses transition prob and emission prob
# beta for the last time point is == 1
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

    



def tuple_converter(s):
    return tuple(int(i) for i in s[1:-1].split(','))

if __name__ == '__main__':
    main()
