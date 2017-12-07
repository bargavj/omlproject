import project
import copy

architectures = ["MLP", "CNN"]
thresholds = [0, 0.01, 0.05, 0.1]
algorithms = ["Adam", "SGD"]
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
				print("\n\n=================== * === * === * ======================")
				print("Calling with: " + arch + " " +  str(threshold) + " " + algo + " " +mode + "\n")

				# INCASE OF POSTTRAIN, we don't need to train the model over and over again:
				if(not(mode == 'posttrain' and threshold != 0)):
					modelCopy = copy.deepcopy(model)
				
				TrainingAccuracies, sparselevel, testaccuracy = project.main(modelCopy, algo, mode, threshold)
				
				print("====================== * === * === * =======================\n\n")

				results.save(arch, algo, mode, threshold, sparselevel, testaccuracy, TrainingAccuracies)

project.main();

f = open('outputData.json', 'w')
f.write(str(results.resultsData))
f.close()