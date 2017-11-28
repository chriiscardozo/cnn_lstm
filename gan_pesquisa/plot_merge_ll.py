import sys, os, csv
import matplotlib.pyplot as plt

if(len(sys.argv) < 2):
	print("Missing result folders parameters")
	exit(0)

plt.clf()
plt.title("GAN MNIST\nNegative Log-Likelihood Comparison per epoch (log-scalar scale)")
plt.ylabel('Negative Log-Likelihood')
plt.xlabel('epoch')
plt.yscale('log')

legends = []
x = [x for x in range(0, 25001, 250)]

for folder in sys.argv[1:]:
	with open(os.path.join(folder, 'lls_avg.csv')) as f:
		reader = csv.reader(f)
		lls_avg = next(reader)
		line, = plt.plot(x, [-float(x) for x in lls_avg], label=folder.replace('/',''))
		legends.append(line)

plt.legend(handles=legends)
plt.savefig('ll_comparison.png')
# plt.show()