import matplotlib.pyplot as plt
from pathlib import Path
import cdh23.loadData
import cdh23.analyzeData
import os

#extracts neurons from PCs for a stack of mice
def main() : 
    #make sure tor ename output directory accordingly
    #make sure to rename output directory accordingly
    #mice = ['m900', 'm722', 'm659', 'm664', 'm666'] #F1
    #mice = ['m1318', 'm1323', 'm609', 'm674', 'm1493'] #ahl
    mice = ['m602', 'm984', 'm985', 'm1315', 'm1321', 'm603', 'm670', 'm675'] #Het
    

    data = cdh23.loadData.loadTheData(mice)
    ordered = data.stackMice()
    analyzer = cdh23.analyzeData.analyzeTheData(ordered[0], ordered[1], "combinedSpace", "het")
    
    output_dir = Path("/Volumes/Data2/Travis/Cdh23/Sean/outputs/het/combinedSpace/neurons")

    PC_dict = [5,10,3,2,4]#[4,6,13,3,5]#[2, 11, 6, 8, 7]
                 

    tones = ['4 kHz', '8 kHz', '16 kHz', '32 kHz', '64 kHz']
    tracker = 0
    for pc in PC_dict :
        #mouseInd = mouseInd + 1
        analyzer.getFeature_Neurons(pc)
        isExist = os.path.exists(Path(output_dir, tones[tracker]))
        if not isExist:
        # Create a new directory because it does not exist
            os.makedirs(Path(output_dir, tones[tracker]))
        plt.savefig(Path(output_dir, tones[tracker]))
        tracker = tracker + 1
            
if __name__ == "__main()__" :
    main()
main()