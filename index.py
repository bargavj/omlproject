import project
import copy
import matplotlib.pyplot as plt
import pickle

architectures = ["MLP", "CNN"]
thresholds = [0, 0.01, 0.025, 0.05, 0.1]
algorithms = ["Adam", "SGD", "Adamax", "RMSprop"]
modes = ["intrain", "posttrain"]


class resultsClass():
	def __init__(self):
		#Initialize the DS
		print("Initializing DS ...")
		self.resultsData = {}
		for arch in architectures:
			self.resultsData[arch] = {}
			for algo in algorithms:
				self.resultsData[arch][algo] = {}
				for mode in modes:
					self.resultsData[arch][algo][mode] = {}
					for threshold in thresholds:
						self.resultsData[arch][algo][mode][threshold] = {"TrainingAccuracies": [], "sparselevel": 0, "testaccuracy": 0}


	def save(self, arch, algo, mode, threshold, sparselevel, testaccuracy, TrainingAccuracies):

		self.resultsData[arch][algo][mode][threshold]['sparselevel'] = sparselevel
		self.resultsData[arch][algo][mode][threshold]['testaccuracy'] = testaccuracy
		self.resultsData[arch][algo][mode][threshold]['TrainingAccuracies'] = TrainingAccuracies
		# print("Saved:")
		# print(str(self.resultsData))



results = resultsClass();



##Main
for arch in architectures:
	model = project.getModel(arch);
	for algo in algorithms:
		for mode in modes:
			for threshold in thresholds:
				if(mode == "intrain" and threshold == 0):
					continue

				print("\n\n=================== * === * === * ======================")
				print("Calling with: " + arch + " " +  str(threshold) + " " + algo + " " +mode + "\n")

				# INCASE OF POSTTRAIN, we don't need to train the model over and over again:
				if(not(mode == 'posttrain' and threshold != 0)):
					modelCopy = copy.deepcopy(model)
				
				TrainingAccuracies, sparselevel, testaccuracy = project.main(modelCopy, algo, mode, threshold)
				
				print("====================== * === * === * =======================\n\n")

				results.save(arch, algo, mode, threshold, sparselevel, testaccuracy, TrainingAccuracies)

				f = open('outputData.json', 'w')
				f.write(str(results.resultsData))
				f.close()


				pickle.dump(results.resultsData, open('outputPickleData.p', "wb"))



# SECTION 1 Plotting Comparisions of all functions:
adamLine = results.resultsData["MLP"]["Adam"]["posttrain"][0]['TrainingAccuracies']
SGD = results.resultsData["MLP"]["SGD"]["posttrain"][0]['TrainingAccuracies']
Adamax = results.resultsData["MLP"]["Adamax"]["posttrain"][0]['TrainingAccuracies']
RMSProp = results.resultsData["MLP"]["RMSprop"]["posttrain"][0]['TrainingAccuracies']
plt.plot(range(1,len(adamLine)+1), adamLine, range(1,len(adamLine)+1), SGD, range(1,len(adamLine)+1), Adamax, range(1,len(adamLine)+1), RMSProp)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('01a-MLPComparision.png')


adamLine = results.resultsData["CNN"]["Adam"]["posttrain"][0]['TrainingAccuracies']
SGD = results.resultsData["CNN"]["SGD"]["posttrain"][0]['TrainingAccuracies']
Adamax = results.resultsData["CNN"]["Adamax"]["posttrain"][0]['TrainingAccuracies']
RMSProp = results.resultsData["CNN"]["RMSprop"]["posttrain"][0]['TrainingAccuracies']
plt.plot(range(1,len(adamLine)+1), adamLine, range(1,len(adamLine)+1), SGD, range(1,len(adamLine)+1), Adamax, range(1,len(adamLine)+1), RMSProp)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('01b-CNNComparision.png')


# SECTION 2 Plotting PostTrain Values for various threshold levels:
AdamLine = []
SGDLine = []
AdamaxLine = []
RMSPropLine = []
AdamX = []
SGDX = []
AdamaxX = []
RMSPropX = []
for threshold in thresholds:
	AdamLine.append(results.resultsData["MLP"]["Adam"]["posttrain"][threshold]["testaccuracy"])
	SGDLine.append(results.resultsData["MLP"]["SGD"]["posttrain"][threshold]["testaccuracy"])
	AdamaxLine.append(results.resultsData["MLP"]["Adamax"]["posttrain"][threshold]["testaccuracy"])
	RMSPropLine.append(results.resultsData["MLP"]["RMSprop"]["posttrain"][threshold]["testaccuracy"])
	AdamX.append(results.resultsData["MLP"]["Adam"]["posttrain"][threshold]["sparselevel"])
	SGDX.append(results.resultsData["MLP"]["SGD"]["posttrain"][threshold]["sparselevel"])
	AdamaxX.append(results.resultsData["MLP"]["AdamaxY"]["posttrain"][threshold]["sparselevel"])
	RMSPropX.append(results.resultsData["MLP"]["RMSprop"]["posttrain"][threshold]["sparselevel"])

plt.plot(AdamX, AdamLine, SGDX, SGDLine, AdamaxX, AdamaxLine, RMSPropX, RMSPropLine)
plt.xlabel('Sparse Level')
plt.ylabel('Accuracy')
plt.savefig('02a-MLPSparseVsAccuracy.png')



AdamLine = []
SGDLine = []
AdamaxLine = []
RMSPropLine = []
AdamX = []
SGDX = []
AdamaxX = []
RMSPropX = []
for threshold in thresholds:
	AdamLine.append(results.resultsData["CNN"]["Adam"]["posttrain"][threshold]["testaccuracy"])
	SGDLine.append(results.resultsData["CNN"]["SGD"]["posttrain"][threshold]["testaccuracy"])
	AdamaxLine.append(results.resultsData["CNN"]["Adamax"]["posttrain"][threshold]["testaccuracy"])
	RMSPropLine.append(results.resultsData["CNN"]["RMSprop"]["posttrain"][threshold]["testaccuracy"])
	AdamX.append(results.resultsData["CNN"]["Adam"]["posttrain"][threshold]["sparselevel"])
	SGDX.append(results.resultsData["CNN"]["SGD"]["posttrain"][threshold]["sparselevel"])
	AdamaxX.append(results.resultsData["CNN"]["AdamaxY"]["posttrain"][threshold]["sparselevel"])
	RMSPropX.append(results.resultsData["CNN"]["RMSprop"]["posttrain"][threshold]["sparselevel"])

plt.plot(AdamX, AdamLine, SGDX, SGDLine, AdamaxX, AdamaxLine, RMSPropX, RMSPropLine)
plt.xlabel('Sparse Level')
plt.ylabel('Accuracy')
plt.savefig('02b-CNNSparseVsAccuracy.png')



