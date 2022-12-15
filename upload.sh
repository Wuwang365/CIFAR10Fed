# scp -r noniid_set dell@10.68.139.146:/home/dell/Desktop/lxc/emnist
# 上传命令
# scp -r ./cpt/cpt_num_* dell@10.68.139.146:/home/dell/Desktop/lxc/emnist/cpt
# 下载命令
# scp -r dell@10.68.139.146:/home/dell/Desktop/lxc/emnist/cpt/*.json /media/wuwang/EXTERNAL_USB/mobicom_23/EMNIST/cpt
scp -r dell@10.68.139.146:/home/dell/Desktop/lxc/emnist/cpt/*.json ./cpt

