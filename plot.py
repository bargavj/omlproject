import json

results = json.load(open('outputDataBackup.json'))

# print(str(results))

adamLine = results["MLP"]["Adam"]["posttrain"]['0']['TrainingAccuracies']
SGD = results["MLP"]["SGD"]["posttrain"]['0']['TrainingAccuracies']
Adamax = results["MLP"]["Adamax"]["posttrain"]['0']['TrainingAccuracies']
RMSProp = results["MLP"]["RMSprop"]["posttrain"]['0']['TrainingAccuracies']
print(str(adamLine))
print(str(SGD))
print(str(Adamax))
print(str(RMSProp))
plt.plot(range(1,len(adamLine)+1), adamLine, range(1,len(adamLine)+1), SGD, range(1,len(adamLine)+1), Adamax, range(1,len(adamLine)+1), RMSProp)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('test1.png')