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
    correct=0
    incorrect=0  
    count_prooved=0
 






    """ test_classes = ["0", "1", "2", "3", "4"]
    correct =0
    incorrect=0
    for o in range(len(test_classes)):
        for i in range(len(test_labels)):
            if test_labels[i] == o:
                if predictions[i] == test_labels[i]:
                    correct+=1
                else:
                    incorrect+=1

    print("Korrekt in Klasse",test_classes[o], "sind", correct)
    print("Inkorrekt in der Klasse",test_classes[o], "sind",incorrect)
    count_prooved=correct+incorrect
    print("Das entspricht einer Genaugigkeit der Klasse", test_classes[o], "von", 100*(correct/count_prooved),"%")
    correct=0
    incorrect=0
    count_prooved=0



    correct =0
incorrect=0
for o in range(len(class_names)):
  correction_class=class_names[o]
  for i in range(len(test_labels)):
    if test_labels[i] == class_names[o]:
      if predictions[i] == test_labels[i]:
        correct+=1
      else:
        incorrect+=1

  
  print("Korrekt in Klasse",class_names[o], "sind", correct)
  print("Inkorrekt in der Klasse",class_names[o], "sind",incorrect)
  count_prooved=correct+incorrect
  print("Das entspricht einer Genaugigkeit der Klasse", class_names[o], "von", 100*(correct/count_prooved),"%")
  correct=0
  incorrect=0  
  count_prooved=0 """