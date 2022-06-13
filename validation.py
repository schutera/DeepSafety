def onceaccurancy(test_labels, predictions, testingclass):
    correct =0
    incorrect=0
    print("================================")
    print("Es wird die genauigkeit der Klasse", testingclass, "ausgegeben.")
    for i in range(len(test_labels)):
        if test_labels[i] == testingclass:
            if predictions[i] == test_labels[i]:
                correct+=1
            else:
                incorrect+=1

  
    print("Korrekt in Klasse",testingclass, "sind", correct)
    print("Inkorrekt in der Klasse", "sind",incorrect)
    count_prooved=correct+incorrect
    print("Das entspricht einer Genaugigkeit der Klasse", testingclass, "von", 100*(correct/count_prooved),"%")
    
 





