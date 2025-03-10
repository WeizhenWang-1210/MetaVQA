import os
import glob

data_dir = "d:/normals"
sdc_files = glob.glob(os.path.join(data_dir, "*.pkl"))

with open("d:/selected_names.txt","r") as f:
    ids = f.read().splitlines()
print(ids)
print(sdc_files)

#selected_sdc_files = [filename for id, filename in zip(ids, sdc_files) if filename.find(id)!=-1]
#print(selected_sdc_files)
selected_files = []
for id in ids:
    for file in sdc_files:
        if file.find(id)!=-1:
            print(file,id)
            selected_files.append(file)
os.makedirs("d:/selected_normals", exist_ok=True)
import shutil
for selected_file in selected_files:
    base = os.path.basename(selected_file)
    shutil.copy(selected_file,f"d:/selected_normals/{base}")
