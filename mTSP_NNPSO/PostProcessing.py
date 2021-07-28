import numpy as np
import matplotlib.pyplot as plt

glob_convergence_file = "/home/supreet/NeuralPSO/mTSP_NNPSO/results/1000_tasks/GlobalConvergence.txt"
mean_convergence_file = "/home/supreet/NeuralPSO/mTSP_NNPSO/results/1000_tasks/MeanConvergence.txt"

GlobalData = np.loadtxt(fname = glob_convergence_file)
MeanData = np.loadtxt(fname = mean_convergence_file)

print(GlobalData)
print(MeanData)

plt.figure(1)
plt.plot(GlobalData[:,0],label="Global Best Function Evaluation")
plt.plot(MeanData[:,0],label="Mean Function Evaluation")
plt.legend(loc="upper right")
plt.xlabel("Iterations")
plt.ylabel("Function Evaluation")

# plt.figure(2)
# plt.plot(GlobalData[:,1],label="Global Best Constraint Violation")
# plt.plot(MeanData[:,1],label="Mean Constraint Violation")
# plt.legend(loc="upper right")
# plt.xlabel("Iterations")
# plt.ylabel("Constraint Violation")

plt.waitforbuttonpress(1000)
