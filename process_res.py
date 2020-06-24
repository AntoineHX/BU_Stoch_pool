import numpy as np
import json, math, time, os

if __name__ == "__main__":
#Res print

    nb_run=3
    accs = []
    taccs = []
    # aug_accs = []
    # f1_max = []
    # f1_min = []
    # times = []
    # mem = []

    files = ["res/benchmark_NoCeil/log/MyLeNetMatStochBUNoceil-50epochs__k4_%s.json"%(str(run)) for run in range(1, nb_run+1)]

    for idx, file in enumerate(files):
        #legend+=str(idx)+'-'+file+'\n'
        with open(file) as json_file:
            data = json.load(json_file)
        # accs.append(data['Accuracy'])
        accs.append(max([x["test_acc"] for x in data]))
        taccs.append(max([x["train_acc"] for x in data]))
        # aug_accs.append(data['Aug_Accuracy'][1])
        # times.append(data['Time'][0])
        # mem.append(data['Memory'][1])
    
        # acc_idx = [x['acc'] for x in data['Log']].index(data['Accuracy'])
        # f1_max.append(max(data['Log'][acc_idx]['f1'])*100)
        # f1_min.append(min(data['Log'][acc_idx]['f1'])*100)
        # print(idx, accs[-1])

    print(files[0])
    print("Acc : %.2f ~ %.2f"%(np.mean(accs), np.std(accs)))
    print("Acc train : %.2f ~ %.2f"%(np.mean(taccs), np.std(taccs)))
    # print("Acc : %.2f ~ %.2f / Aug_Acc %d: %.2f ~ %.2f"%(np.mean(accs), np.std(accs), data['Aug_Accuracy'][0], np.mean(aug_accs), np.std(aug_accs)))
    # print("F1 max : %.2f ~ %.2f / F1 min : %.2f ~ %.2f"%(np.mean(f1_max), np.std(f1_max), np.mean(f1_min), np.std(f1_min)))
    # print("Time (h): %.2f ~ %.2f"%(np.mean(times)/3600, np.std(times)/3600))
    # print("Mem (MB): %d ~ %d"%(np.mean(mem), np.std(mem)))