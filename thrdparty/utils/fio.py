

def writelines(list,path):
    lenth = len(list)
    with open(path,'w') as f:
        for i in range(lenth):
            if i == lenth-1:
                f.writelines(str(list[i]))
            else:
                f.writelines(str(list[i])+'\n')

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def split2files(data_path,split_txt,dataset=None):
    files=[]
    lines = readlines(split_txt)
    for line in lines:
        files.append(str(data_path)+'/'+line)
    return files