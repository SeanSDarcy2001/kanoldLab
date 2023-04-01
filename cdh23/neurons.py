import matplotlib.pyplot as plt
from pathlib import Path
import cdh23.loadData
import cdh23.analyzeData
import os

def main() : 
    #make sure tor ename output directory accordingly
    #mice = ['m659', 'm664', 'm666', 'm722', 'm900'] #F1 Ahl/ahl somewthing off with 722 and 900
    #mice = ['m602', 'm603', 'm670', 'm675', 'm984', 'm985', 'm1315', 'm1321'] #Ahl/ahl
    #mice = ['m1493'] #ahl/ahl #'m609', 'm674', 'm1318',

    mice = ['m659', 'm664', 'm666']
    output_dir = Path("/Volumes/Data2/Travis/Cdh23/Sean/outputs/neurons")
    data = cdh23.loadData.loadTheData(mice)

    PC_dict = {mice[0]: [4, 5, 6, 8, 7],
                 mice[1]: [6, 4, 7, 4, 12],
                 mice[2]: [8, 7, 4, 3, 6]}
    

    mouseInd = 0
    for mouse in mice :
        print("----------------")
        print("ANALYZING MOUSE:", mouse)
        print("----------------")
        ordered = data.processAndSort(mouse)
        analyzer = cdh23.analyzeData.analyzeTheData(ordered[0], ordered[1], mouse)
        tones = ['4 kHz', '8 kHz', '16 kHz', '32 kHz', '64 kHz']
        tracker = 0
        for pc in PC_dict[mouse] :
            mouseInd = mouseInd + 1
            analyzer.getFeature_Neurons(pc)
            isExist = os.path.exists(Path(output_dir, mouse, tones[tracker]))
            if not isExist:
            # Create a new directory because it does not exist
                os.makedirs(Path(output_dir, mouse, tones[tracker]))
            plt.savefig(Path(output_dir, mouse, tones[tracker]))
            tracker = tracker + 1
            
if __name__ == "__main()__" :
    main()
main()