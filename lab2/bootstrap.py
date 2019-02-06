import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np


def boostrap(sample, sample_size, iterations):
	# <---INSERT YOUR CODE HERE--->
	samples  = np.random.choice(data,replace = True, size = [iterations, len(data)])
	#print samples.shape
	data_mean = data.mean()
	values = []
	for sample in samples:
		#print sample
		values.append(sample)
	temp = np.array(values)
	#print temp
	lower, upper = np.percentile(temp, [5, 95])
	return data_mean,lower, upper


if __name__ == "__main__":
	df = pd.read_csv('lab2/salaries.csv')

	data = df.values.T[1]
	boots = []
	for i in range(100, 100000, 1000):
		boot = boostrap(data, data.shape[0], i)
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])
		print(boots)

	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

	sns_plot.axes[0, 0].set_ylim(0,)
	sns_plot.axes[0, 0].set_xlim(0, 100000)

	sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')


	#print ("Mean: %f")%(np.mean(data))
	#print ("Var: %f")%(np.var(data))
	


	