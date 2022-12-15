import torch
import glob
from matplotlib import pyplot as plt
import numpy as np
import re
from visual.figure_parameter import (
    font_style,
    title_style
)


cpt_names = glob.glob("./cpt_char/*.pth")
cpt_names.sort(key=lambda x:int("".join(re.findall("\d",x))))
all_cpt = []

for cpt in cpt_names:
    weight = torch.load(cpt)
    for parameter in weight.parameters():
        cur_cpt = torch.mean(parameter,dim=[1,2,3]).cpu().detach().numpy()
        break
    all_cpt.append(cur_cpt)
    
first_weight = all_cpt[0]
all_cpt = np.asarray(all_cpt)[1:]

col = np.shape(all_cpt)[1]
row = np.shape(all_cpt)[0]

for i in range(row):
    all_cpt[i] = np.abs(all_cpt[i]-first_weight)

for i in range(col):
    # all_cpt[i] = np.abs(all_cpt[i]-first_weight)
    sub_min = all_cpt[:,i]-np.min(all_cpt[:,i])
    
    # all_cpt[:,i] = sub_min/np.max(sub_min)
# all_cpt = all_cpt[1:]/np.max(all_cpt)

x_ticks = np.arange(-0.5,63.5,1)
y_ticks = np.arange(-0.5,len(cpt_names)-0.5,1)
x_ticks_display = np.arange(0,64)
y_ticks_display = np.arange(0,len(cpt_names))

fig = plt.figure()
ax = plt.subplot()

ax.set_xticks(x_ticks)
ax.set_xticklabels([])
ax.set_yticks(y_ticks)
ax.set_yticklabels([])


cax = ax.imshow(all_cpt,"bwr")
cbar = plt.colorbar(cax, drawedges = False)

plt.title("Character Classification",fontdict=title_style)
plt.xlabel("Filter index",fontdict=font_style)
plt.ylabel("Weight index",fontdict=font_style)

ax.grid()
# plt.show()
# plt.waitforbuttonpress()
plt.savefig("./char.svg")