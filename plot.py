import pickle
import matplotlib.pyplot as plt

results = pickle.load(open('outputPickleData.p', 'rb'))

###### MLP Accuracy w/o sparsity ######
Adam = results["MLP"]["Adam"]["posttrain"][0]['TrainingAccuracies']
SGD = results["MLP"]["SGD"]["posttrain"][0]['TrainingAccuracies']
Adamax = results["MLP"]["Adamax"]["posttrain"][0]['TrainingAccuracies']
RMSprop = results["MLP"]["RMSprop"]["posttrain"][0]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('MLP_acc_nonsparse.png')
plt.close()

###### CNN Accuracy w/o sparsity ######
Adam = results["CNN"]["Adam"]["posttrain"][0]['TrainingAccuracies']
SGD = results["CNN"]["SGD"]["posttrain"][0]['TrainingAccuracies']
Adamax = results["CNN"]["Adamax"]["posttrain"][0]['TrainingAccuracies']
RMSprop = results["CNN"]["RMSprop"]["posttrain"][0]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('CNN_acc_nonsparse.png')
plt.close()

###### MLP Accuracy posttrain sparsity ######
AdamSparsity = [results["MLP"]["Adam"]["posttrain"][i]['sparselevel'] for i in [0, 0.01, 0.025, 0.05, 0.1]]
AdamAccuracy = [results["MLP"]["Adam"]["posttrain"][i]['testaccuracy'] for i in [0, 0.01, 0.025, 0.05, 0.1]]

SGDSparsity = [results["MLP"]["SGD"]["posttrain"][i]['sparselevel'] for i in [0, 0.01, 0.025, 0.05, 0.1]]
SGDAccuracy = [results["MLP"]["SGD"]["posttrain"][i]['testaccuracy'] for i in [0, 0.01, 0.025, 0.05, 0.1]]

AdamaxSparsity = [results["MLP"]["Adamax"]["posttrain"][i]['sparselevel'] for i in [0, 0.01, 0.025, 0.05, 0.1]]
AdamaxAccuracy = [results["MLP"]["Adamax"]["posttrain"][i]['testaccuracy'] for i in [0, 0.01, 0.025, 0.05, 0.1]]

RMSpropSparsity = [results["MLP"]["RMSprop"]["posttrain"][i]['sparselevel'] for i in [0, 0.01, 0.025, 0.05, 0.1]]
RMSpropAccuracy = [results["MLP"]["RMSprop"]["posttrain"][i]['testaccuracy'] for i in [0, 0.01, 0.025, 0.05, 0.1]]

plt.plot(AdamSparsity, AdamAccuracy, 'r-+', label='Adam')
plt.plot(SGDSparsity, SGDAccuracy, 'b-', label='SGD')
plt.plot(AdamaxSparsity, AdamaxAccuracy, 'g-.', label='Adamax')
plt.plot(RMSpropSparsity, RMSpropAccuracy, '-.', label='RMSprop')
plt.xlabel('Sparsity')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.savefig('MLP_acc_posttrain_sparse.png')
plt.close()

###### CNN Accuracy posttrain sparsity ######
AdamSparsity = [results["CNN"]["Adam"]["posttrain"][i]['sparselevel'] for i in [0, 0.01, 0.025, 0.05, 0.1]]
AdamAccuracy = [results["CNN"]["Adam"]["posttrain"][i]['testaccuracy'] for i in [0, 0.01, 0.025, 0.05, 0.1]]

SGDSparsity = [results["CNN"]["SGD"]["posttrain"][i]['sparselevel'] for i in [0, 0.01, 0.025, 0.05, 0.1]]
SGDAccuracy = [results["CNN"]["SGD"]["posttrain"][i]['testaccuracy'] for i in [0, 0.01, 0.025, 0.05, 0.1]]

AdamaxSparsity = [results["CNN"]["Adamax"]["posttrain"][i]['sparselevel'] for i in [0, 0.01, 0.025, 0.05, 0.1]]
AdamaxAccuracy = [results["CNN"]["Adamax"]["posttrain"][i]['testaccuracy'] for i in [0, 0.01, 0.025, 0.05, 0.1]]

RMSpropSparsity = [results["CNN"]["RMSprop"]["posttrain"][i]['sparselevel'] for i in [0, 0.01, 0.025, 0.05, 0.1]]
RMSpropAccuracy = [results["CNN"]["RMSprop"]["posttrain"][i]['testaccuracy'] for i in [0, 0.01, 0.025, 0.05, 0.1]]

plt.plot(AdamSparsity, AdamAccuracy, 'r-+', label='Adam')
plt.plot(SGDSparsity, SGDAccuracy, 'b-', label='SGD')
plt.plot(AdamaxSparsity, AdamaxAccuracy, 'g-.', label='Adamax')
plt.plot(RMSpropSparsity, RMSpropAccuracy, '-.', label='RMSprop')
plt.xlabel('Sparsity')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.savefig('CNN_acc_posttrain_sparse.png')
plt.close()

###### MLP Accuracy intrain sparsity ######
AdamSparsity = [results["MLP"]["Adam"]["intrain"][i]['sparselevel'] for i in [0.01, 0.025, 0.05, 0.1]]
AdamSparsity.insert(0, results["MLP"]["Adam"]["posttrain"][0]['sparselevel'])
AdamAccuracy = [results["MLP"]["Adam"]["intrain"][i]['testaccuracy'] for i in [0.01, 0.025, 0.05, 0.1]]
AdamAccuracy.insert(0, results["MLP"]["Adam"]["posttrain"][0]['testaccuracy'])

SGDSparsity = [results["MLP"]["SGD"]["intrain"][i]['sparselevel'] for i in [0.01, 0.025, 0.05, 0.1]]
SGDSparsity.insert(0, results["MLP"]["SGD"]["posttrain"][0]['sparselevel'])
SGDAccuracy = [results["MLP"]["SGD"]["intrain"][i]['testaccuracy'] for i in [0.01, 0.025, 0.05, 0.1]]
SGDAccuracy.insert(0, results["MLP"]["SGD"]["posttrain"][0]['testaccuracy'])

AdamaxSparsity = [results["MLP"]["Adamax"]["intrain"][i]['sparselevel'] for i in [0.01, 0.025, 0.05, 0.1]]
AdamaxSparsity.insert(0, results["MLP"]["Adamax"]["posttrain"][0]['sparselevel'])
AdamaxAccuracy = [results["MLP"]["Adamax"]["intrain"][i]['testaccuracy'] for i in [0.01, 0.025, 0.05, 0.1]]
AdamaxAccuracy.insert(0, results["MLP"]["Adamax"]["posttrain"][0]['testaccuracy'])

RMSpropSparsity = [results["MLP"]["RMSprop"]["intrain"][i]['sparselevel'] for i in [0.01, 0.025, 0.05, 0.1]]
RMSpropSparsity.insert(0, results["MLP"]["RMSprop"]["posttrain"][0]['sparselevel'])
RMSpropAccuracy = [results["MLP"]["RMSprop"]["intrain"][i]['testaccuracy'] for i in [0.01, 0.025, 0.05, 0.1]]
RMSpropAccuracy.insert(0, results["MLP"]["RMSprop"]["posttrain"][0]['testaccuracy'])

plt.plot(AdamSparsity, AdamAccuracy, 'r-+', label='Adam')
plt.plot(SGDSparsity, SGDAccuracy, 'b-', label='SGD')
plt.plot(AdamaxSparsity, AdamaxAccuracy, 'g-.', label='Adamax')
plt.plot(RMSpropSparsity, RMSpropAccuracy, '-.', label='RMSprop')
plt.xlabel('Sparsity')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.savefig('MLP_acc_intrain_sparse.png')
plt.close()

###### CNN Accuracy intrain sparsity ######
AdamSparsity = [results["CNN"]["Adam"]["intrain"][i]['sparselevel'] for i in [0.01, 0.025, 0.05, 0.1]]
AdamSparsity.insert(0, results["CNN"]["Adam"]["posttrain"][0]['sparselevel'])
AdamAccuracy = [results["CNN"]["Adam"]["intrain"][i]['testaccuracy'] for i in [0.01, 0.025, 0.05, 0.1]]
AdamAccuracy.insert(0, results["CNN"]["Adam"]["posttrain"][0]['testaccuracy'])

SGDSparsity = [results["CNN"]["SGD"]["intrain"][i]['sparselevel'] for i in [0.01, 0.025, 0.05, 0.1]]
SGDSparsity.insert(0, results["CNN"]["SGD"]["posttrain"][0]['sparselevel'])
SGDAccuracy = [results["CNN"]["SGD"]["intrain"][i]['testaccuracy'] for i in [0.01, 0.025, 0.05, 0.1]]
SGDAccuracy.insert(0, results["CNN"]["SGD"]["posttrain"][0]['testaccuracy'])

AdamaxSparsity = [results["CNN"]["Adamax"]["intrain"][i]['sparselevel'] for i in [0.01, 0.025, 0.05, 0.1]]
AdamaxSparsity.insert(0, results["CNN"]["Adamax"]["posttrain"][0]['sparselevel'])
AdamaxAccuracy = [results["CNN"]["Adamax"]["intrain"][i]['testaccuracy'] for i in [0.01, 0.025, 0.05, 0.1]]
AdamaxAccuracy.insert(0, results["CNN"]["Adamax"]["posttrain"][0]['testaccuracy'])

RMSpropSparsity = [results["CNN"]["RMSprop"]["intrain"][i]['sparselevel'] for i in [0.01, 0.025, 0.05, 0.1]]
RMSpropSparsity.insert(0, results["CNN"]["RMSprop"]["posttrain"][0]['sparselevel'])
RMSpropAccuracy = [results["CNN"]["RMSprop"]["intrain"][i]['testaccuracy'] for i in [0.01, 0.025, 0.05, 0.1]]
RMSpropAccuracy.insert(0, results["CNN"]["RMSprop"]["posttrain"][0]['testaccuracy'])

plt.plot(AdamSparsity, AdamAccuracy, 'r-+', label='Adam')
plt.plot(SGDSparsity, SGDAccuracy, 'b-', label='SGD')
plt.plot(AdamaxSparsity, AdamaxAccuracy, 'g-.', label='Adamax')
plt.plot(RMSpropSparsity, RMSpropAccuracy, '-.', label='RMSprop')
plt.xlabel('Sparsity')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.savefig('CNN_acc_intrain_sparse.png')
plt.close()

###### MLP Accuracy intrain 0.01 sparsity ######
Adam = results["MLP"]["Adam"]["intrain"][0.01]['TrainingAccuracies']
SGD = results["MLP"]["SGD"]["intrain"][0.01]['TrainingAccuracies']
Adamax = results["MLP"]["Adamax"]["intrain"][0.01]['TrainingAccuracies']
RMSprop = results["MLP"]["RMSprop"]["intrain"][0.01]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('MLP_acc_intrain_0.01.png')
plt.close()

###### CNN Accuracy intrain 0.01 sparsity ######
Adam = results["CNN"]["Adam"]["intrain"][0.01]['TrainingAccuracies']
SGD = results["CNN"]["SGD"]["intrain"][0.01]['TrainingAccuracies']
Adamax = results["CNN"]["Adamax"]["intrain"][0.01]['TrainingAccuracies']
RMSprop = results["CNN"]["RMSprop"]["intrain"][0.01]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('CNN_acc_intrain_0.01.png')
plt.close()

###### MLP Accuracy intrain 0.025 sparsity ######
Adam = results["MLP"]["Adam"]["intrain"][0.025]['TrainingAccuracies']
SGD = results["MLP"]["SGD"]["intrain"][0.025]['TrainingAccuracies']
Adamax = results["MLP"]["Adamax"]["intrain"][0.025]['TrainingAccuracies']
RMSprop = results["MLP"]["RMSprop"]["intrain"][0.025]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('MLP_acc_intrain_0.025.png')
plt.close()

###### CNN Accuracy intrain 0.025 sparsity ######
Adam = results["CNN"]["Adam"]["intrain"][0.025]['TrainingAccuracies']
SGD = results["CNN"]["SGD"]["intrain"][0.025]['TrainingAccuracies']
Adamax = results["CNN"]["Adamax"]["intrain"][0.025]['TrainingAccuracies']
RMSprop = results["CNN"]["RMSprop"]["intrain"][0.025]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('CNN_acc_intrain_0.025.png')
plt.close()

###### MLP Accuracy intrain 0.05 sparsity ######
Adam = results["MLP"]["Adam"]["intrain"][0.05]['TrainingAccuracies']
SGD = results["MLP"]["SGD"]["intrain"][0.05]['TrainingAccuracies']
Adamax = results["MLP"]["Adamax"]["intrain"][0.05]['TrainingAccuracies']
RMSprop = results["MLP"]["RMSprop"]["intrain"][0.05]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('MLP_acc_intrain_0.05.png')
plt.close()

###### CNN Accuracy intrain 0.05 sparsity ######
Adam = results["CNN"]["Adam"]["intrain"][0.05]['TrainingAccuracies']
SGD = results["CNN"]["SGD"]["intrain"][0.05]['TrainingAccuracies']
Adamax = results["CNN"]["Adamax"]["intrain"][0.05]['TrainingAccuracies']
RMSprop = results["CNN"]["RMSprop"]["intrain"][0.05]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('CNN_acc_intrain_0.05.png')
plt.close()

###### MLP Accuracy intrain 0.1 sparsity ######
Adam = results["MLP"]["Adam"]["intrain"][0.1]['TrainingAccuracies']
SGD = results["MLP"]["SGD"]["intrain"][0.1]['TrainingAccuracies']
Adamax = results["MLP"]["Adamax"]["intrain"][0.1]['TrainingAccuracies']
RMSprop = results["MLP"]["RMSprop"]["intrain"][0.1]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('MLP_acc_intrain_0.1.png')
plt.close()

###### CNN Accuracy intrain 0.1 sparsity ######
Adam = results["CNN"]["Adam"]["intrain"][0.1]['TrainingAccuracies']
SGD = results["CNN"]["SGD"]["intrain"][0.1]['TrainingAccuracies']
Adamax = results["CNN"]["Adamax"]["intrain"][0.1]['TrainingAccuracies']
RMSprop = results["CNN"]["RMSprop"]["intrain"][0.1]['TrainingAccuracies']
plt.plot(range(1,len(Adam)+1), Adam, 'r-+', label='Adam')
plt.plot(range(1,len(Adam)+1), SGD, 'b-', label='SGD')
plt.plot(range(1,len(Adam)+1), Adamax, 'g-.', label='Adamax')
plt.plot(range(1,len(Adam)+1), RMSprop, '-.', label='RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('CNN_acc_intrain_0.1.png')
plt.close()