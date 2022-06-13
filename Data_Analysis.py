import pyplot

#werte RMS Einlesen
#Accuracy Werte einlesen
#Werte einander zuordnen
#Sortieren und Gruppieren (RMS Abhängig)
#durchschnittl. RMS pro Gruppe berechnen
#"Categorical Plotting"

names = ['group_a', 'group_b', 'group_c', 'group_d', 'group_e']
values = [1, 10, 100, 110, 50]
plt.figure(figsize=(9, 3))

#plt.subplot(131)
#plt.bar(names, values)
#plt.subplot(132)
#plt.scatter(names, values)
subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()