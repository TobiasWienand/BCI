import numpy as np

arr = np.zeros((85, 8))
with open("TSClafs_performances.txt") as f:
    for i, line in enumerate(f.readlines()):
        words = line.split()
        numbers = [float(num) for num in words[1:]]
        arr[i, :] = np.array(numbers)

print(np.mean(arr, axis = 0))
print("Conclusion: ROCKET > ITime > CHIEF > HCTE > ResNet > ST > PF > BOSS")