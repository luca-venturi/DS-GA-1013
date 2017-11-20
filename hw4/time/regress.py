import numpy as np
_beta = np.array([9,1,1,1,1])

"""
Returns the tuple train_X,train_y,valid_X,valid_y containing
the training and validation sets.
"""
def get_data() :
    global _beta
    np.random.seed(1234)
    num_train, num_valid = 50,20
    n = num_train+num_valid
    X = np.zeros((n,5))
    X[:,0] = 1
    X[:,1] = X[:,0]+np.random.randn(n)
    for i in range(2,5) :
        X[:,i] = X[:,i-1] + np.random.randn(n)*0.01
    y = np.dot(X,_beta) + np.random.randn(n)
    return X[:num_train,:],y[:num_train],X[num_train:,:],y[num_train:]

def squareLoss(x,y,b):
	loss = 0.
	for i in range(y.size):
		loss += (np.dot(b,x[i,:]) - y[i]) ** 2
	return loss

def main() :
	n = 5
	train_X,train_y,valid_X,valid_y = get_data()
	N = train_y.size	
	M = np.zeros((n,n))
	v = np.zeros((n))
	for i in range(n):
		M += np.outer(train_X[i,:],train_X[i,:])
		v += train_y[i] * train_X[i,:]
	# least square	
	beta_est = np.linalg.solve(M,v)
	print 'estimated beta: ', beta_est
	print 'train loss: ', squareLoss(train_X,train_y,beta_est)
	print 'test loss: ', squareLoss(valid_X,valid_y,beta_est)
	_, s, _ = np.linalg.svd(train_X, full_matrices=False)
	print 'singular values of training data: ', s
	# ridge regression
	beta_est = np.linalg.solve(M + 0.5 * np.eye(n),v)
	print 'RR estimated beta: ', beta_est
	print 'RR train loss: ', squareLoss(train_X,train_y,beta_est)
	print 'RR test loss: ', squareLoss(valid_X,valid_y,beta_est)
	

if __name__ == "__main__" :
    main()
