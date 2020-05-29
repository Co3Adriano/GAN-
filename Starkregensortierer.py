##DWD ASC Datenbank
import os
import shutil
root_dir = 'E:/BachelorArbeit/KI_Regen/RadoLan_ASC_SET/'  # root_directory hier befinden sich die zu sortierenden Datein

###function für asc datei vom dwd zu datenbank
for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".asc") :
            print (filepath)
            with open(os.path.join(subdir, filename)) as f:
             print(filepath)
             #npArray = np.loadtxt(f,skiprows = 6)
             datensatz = open(filepath).readlines()[6:]            
             for line in datensatz:
                 if '150' in line: # X für Starkregen in 1/mm/h da die Asc file stündlich gegeben
                     print("Starkregen!!!!")
                     shutil.copy(filepath, 'E:/BachelorArbeit/KI_Regen/RadoLan_ASC_SET/Starkregen/') # hier wird die Datei hinkopiert

           




