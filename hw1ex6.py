import numpy as np

M = [[.9113, .2501, -.1344], [-.0124, 1.3625, -.129], [1.0739, 1.5005, -.2738]]
w, v = np.linalg.eig(M)

print w
print v
