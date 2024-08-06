import zipfile
import os
import sys

def zipfolder(foldername, target_dir):            
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])
    zipobj.close()

zipfolder('/pscratch/sd/m/mavaylon/chem_final_data/Traj/Traj_Zip_Final/Final_Chunked_Singlet_Doublet', '/pscratch/sd/m/mavaylon/chem_final_data/Traj/Traj_Zip_Final/Final_Chunked_Singlet_Doublet')
sys.exit()
