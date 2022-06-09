import os, shutil

if __name__ ==  "__main__":
    mode = 'slow_testset'
    # /home/kuanhaochen/Documents/Kings College/Individual_Project/TMNet/datasets/vimeo_septuplet
    inPath = '/home/kuanhaochen/Documents/Kings College/Individual_Project/TMNet/datasets/vimeo_septuplet/sequences/test/'
    outPath = '/home/kuanhaochen/Documents/Kings College/Individual_Project/TMNet/datasets/vimeo_septuplet/slow_of_test/HR/'
    guide = '/home/kuanhaochen/Documents/Kings College/Individual_Project/TMNet/datasets/vimeo_septuplet/sep_' + mode + '.txt'
    # inPath = '/home/ubuntu/Disk/dataset/Vimeo-90k/vimeo_septuplet/sequences/'
    # outPath = '/home/ubuntu/Disk/dataset/Vimeo-90k/vimeo_septuplet/sequences/' + mode + '/'
    # guide = '/home/ubuntu/Disk/dataset/Vimeo-90k/vimeo_septuplet/sep_' + mode + 'list.txt'
    
    f = open(guide, "r")
    lines = f.readlines()
    
    if not os.path.isdir(outPath):
        os.mkdir(outPath)
        # os.makedirs(outPath) # for fast/medium/slow only

    for l in lines:
        line = l.replace('\n','')
        this_folder = os.path.join(inPath, line)
        dest_folder = os.path.join(outPath, line)
        print(this_folder)
        shutil.copytree(this_folder, dest_folder)
    print('Done')
