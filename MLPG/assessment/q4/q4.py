# transitions = {"From":[], "To":[], "Prob":[]}
# emmisions = {"State":[], "Symbol":[], "Prob":[]}
transitions = {}
emmisions = {}
observations = [[]]
states = ['(0,0)','(0,1)','(1,0)','(1,1)']

def main():
    # transitions = np.genfromtxt('transition.txt',names=True, delimiter=' ',encoding='UTF-8',dtype=None)
    # emissions = np.genfromtxt('emmision.txt',names=True, delimiter=' ',encoding='UTF-8',dtype=None)
    
    # for row in range(len(transitions)):
    #     for col in range(2):
    #         if col == 0:
    #             if row > 16:
    #                 transitions[row][col] = tuple(int(i) for i in transitions[row][col][1:-1].split(','))
    #         elif col == 1:
    #             transitions[row][col] = tuple(int(i) for i in transitions[row][col][1:-1].split(','))
    #     print(transitions[row])

    # observations, states, start_prob, trans_prob, emm_prob, end_state


    # with open('q4hmm.txt') as f:
    #     i = 0
    #     data = f.readlines()
    #     for line in data:
    #         i += 1
    #         line = line.rstrip().split(' ')
    #         if i > 2 and i <= 274 :
    #             # Do Transition
    #             transitions['From'].append(line[0]) if i <= 18 else transitions['From'].append(tuple_converter(line[0]))
    #             transitions['To'].append(tuple_converter(line[1]))
    #             transitions['Prob'].append(float(line[2]))
    #         elif i >= 277:
    #             # Do Emmision
    #             emmisions['State'].append(tuple_converter(line[0]))
    #             emmisions['Symbol'].append(int(line[1]))
    #             emmisions['Prob'].append(float(line[2]))
    with open('q4hmm.txt') as f:
        i = 0
        data = f.readlines()
        for line in data:
            i += 1
            line = line.rstrip().split(' ')
            if i > 2 and i <= 274:
                # Do Transition
                if (line[0] not in transitions.keys()):
                    transitions[line[0]] = {}
                transitions[line[0]].update({line[1]: float(line[2])})
            elif i >= 277:
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
        observations.pop()
    
    # print(observations)
    # print(transitions['From'][0])
    # print(transitions['To'][0])
    # print(transitions['Prob'][0])
    # print(emmisions['State'][0])
    # print(emmisions['Symbol'][0])
    # print(emmisions['Prob'][0])

    #print prob for a given state transition
    #print(transitions['(None,None)'][states[0]])

    print(forward(observations[0]))




def forward(obsv):
    forward_prob = []
    h_prev = {}
    for i, obsv_i in enumerate(obsv):
        h_c = {}
        for state in states:
            if i == 0:
                print("Starting state...")
                total_h = transitions['(None,None)'][state]
            else:
                total_h = sum(h_prev[j] * transitions[j][state] for j in states)
            h_c[state] = emmisions[state][obsv_i] * total_h
        forward_prob.append(h_c)
        h_prev = h_c

    return forward_prob
# uses transition prob and emission prob
# beta for the last time point is == 1
def backward(trans_P,emm_P,beta):
    return trans_P * emm_P * beta

# def print_prob(n_seq, length, probabilities):
#     print("Sequence "+n_seq+", length ":length)
#     for state in probabilities.keys():
#         pass



def tuple_converter(s):
    return tuple(int(i) for i in s[1:-1].split(','))

if __name__ == '__main__':
    main()
