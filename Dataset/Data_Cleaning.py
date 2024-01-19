import os
#make every directory have same size

def make_same_size(root_path):
    
    
    root_list=os.listdir(root_path)
    for file in root_list:
        path = os.path.join(root_path, file)
        folder = os.listdir(path)
        for f in folder:
            imgpath = os.path.join(path, f)
            img = os.listdir(imgpath)
            delimg=[]
            while len(img) > 70:
                delimg.append(img[-1])
                os.remove(os.path.join(imgpath, img[-1]))
                img.pop(-1)

train_path='Training'
val_path='Validation'
make_same_size(train_path)
make_same_size(val_path)
