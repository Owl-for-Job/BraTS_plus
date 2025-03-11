import os
from sklearn.model_selection import train_test_split

# 预处理输出地址
data_path = "E:\DataSet\生医图像处理\output"
train_and_test_ids = os.listdir(data_path)

train_ids, val_test_ids = train_test_split(train_and_test_ids, test_size=0.2,random_state=21)
val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5,random_state=21)
print("Using {} images for training, {} images for validation, {} images for testing.".format(len(train_ids),len(val_ids),len(test_ids)))

with open('E:\\DataSet\\生医图像处理\\output\\train.txt','w') as f:
    f.write('\n'.join(train_ids))

with open('E:\\DataSet\\生医图像处理\\output\\valid.txt','w') as f:
    f.write('\n'.join(val_ids))

with open('E:\\DataSet\\生医图像处理\\output\\test.txt','w') as f:
    f.write('\n'.join(test_ids))
