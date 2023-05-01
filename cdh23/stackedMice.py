import cdh23.loadData
import cdh23.analyzeData
#stacks a set of mice and generates data for the stack
def main() : 
    #make sure to rename output directory accordingly
    mice = ['m900', 'm722', 'm659', 'm664', 'm666'] #F1
    #mice = ['m1318', 'm1323', 'm609', 'm674', 'm1493'] #ahl
    #mice = ['m602', 'm984', 'm985', 'm1315', 'm1321', 'm603', 'm670', 'm675'] #Het
    

    data = cdh23.loadData.loadTheData(mice)
    ordered = data.stackMice()
    analyzer = cdh23.analyzeData.analyzeTheData(ordered[0], ordered[1], "newCombined", "F1_Ahl")
    #dataToClassify = analyzer.trial_response_pca()
    #for k in range(3) : #iterate through plots
        #analyzer.classify(.25, k, dataToClassify)
    #analyzer.trial_average_pca()
    #analyzer.threeDtrajectories()
    #analyzer.twoD_vid()
    #analyzer.onsetOffset()
    #analyzer.tsne()
    analyzer.tsne2()
    
            
if __name__ == "__main()__" :
    main()
main()