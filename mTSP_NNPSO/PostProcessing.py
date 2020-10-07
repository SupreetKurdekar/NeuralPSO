import numpy as np
import matplotlib.pyplot as plt

GlobalData = np.loadtxt(fname = "mTSP_NNPSO\GlobalConvergence.txt")
MeanData = np.loadtxt(fname = "mTSP_NNPSO\MeanConvergence.txt")

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
