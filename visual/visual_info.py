from matplotlib import pyplot as plt
import json

def visual_training_info():
    with open("../training_info.json","r") as f:
        json_info = json.loads(f.read())
    acc_array = []
    for key in json_info.keys():
        acc_array.append(json_info[key]["acc"])
    plt.plot(acc_array)
    plt.show()
    
if __name__=="__main__":
    visual_training_info()