"""
Some old util code that might be useful in the future
"""
import torch # To run K-means with GPU

# Modified version of K-Means to add Earth Mover's Distance from here: https://github.com/subhadarship/kmeans_pytorch/blob/master/kmeans_pytorch/__init__.py
# DEPRECATED, using the standard kmeans from scikit-learn since it suits our needs
def initialize(X, n_clusters, seed):
	"""
	initialize cluster centers
	:param X: (torch.tensor) matrix
	:param n_clusters: (int) number of clusters
	:param seed: (int) seed for kmeans
	:return: (np.array) initial state
	"""
	num_samples = len(X)
	if seed == None:
		indices = np.random.choice(num_samples, n_clusters, replace=False)
	else:
		np.random.seed(seed) ; indices = np.random.choice(num_samples, n_clusters, replace=False)
	initial_state = X[indices]
	return initial_state

def kmeans_custom(
		X,
		n_clusters,
		distance='euclidean',
		centroids=[],
		tol=1e-3,
		tqdm_flag=True,
		iter_limit=0,
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
		seed=None,
):
	"""
	perform kmeans n_init, default=10
	Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
	:param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
	Return
		X_cluster_ids (torch.tensor), centroids (torch.tensor)
	"""
	if tqdm_flag:
		print(f'running k-means on {device}..')

	if distance == 'euclidean':
		pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
	elif distance == 'cosine':
		pairwise_distance_function = partial(pairwise_cosine, device=device)
	elif distance == 'EMD':
		pairwise_distance_function = partial(pairwise_EMD, device=device)

	else:
		raise NotImplementedError

	if type(X) != torch.Tensor:
		X = torch.tensor(X)
	# convert to float
	X = X.float()

	# transfer to device
	X = X.to(device)

	# initialize
	if type(centroids) == list:  # ToDo: make this less annoyingly weird
		initial_state = initialize(X, n_clusters, seed=seed)
	else:
		if tqdm_flag:
			print('resuming')
		# find data point closest to the initial cluster center
		initial_state = centroids
		dis = pairwise_distance_function(X, initial_state)
		choice_points = torch.argmin(dis, dim=0)
		initial_state = X[choice_points]
		initial_state = initial_state.to(device)

	iteration = 0
	if tqdm_flag:
		tqdm_meter = tqdm(desc='[running kmeans]')

	while True:
		dis = pairwise_distance_function(X, initial_state)

		choice_cluster = torch.argmin(dis, dim=1)

		initial_state_pre = initial_state.clone()

		for index in range(n_clusters):
			selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

			selected = torch.index_select(X, 0, selected)

			# https://github.com/subhadarship/kmeans_pytorch/issues/16
			if selected.shape[0] == 0:
				selected = X[torch.randint(len(X), (1,))]

			initial_state[index] = selected.mean(dim=0)

		center_shift = torch.sum(
			torch.sqrt(
				torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
			))

		# increment iteration
		iteration = iteration + 1

		# update tqdm meter
		if tqdm_flag:
			tqdm_meter.set_postfix(
				iteration=f'{iteration}',
				center_shift=f'{center_shift ** 2:0.6f}',
				tol=f'{tol:0.6f}'
			)
			tqdm_meter.update()
		if center_shift ** 2 < tol:
			break
		if iter_limit != 0 and iteration >= iter_limit:
			break

	return choice_cluster.cpu(), initial_state.cpu() # clusters_indices_on_initial data, final_centroids

def kmeans_custom_predict(
		X,
		centroids,
		distance='euclidean',
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
		tqdm_flag=True
):
	"""
	Return
		cluster_ids_on_X (torch.tensor)
	
	"""
	if tqdm_flag:
		print(f'predicting on {device}..')

	if distance == 'euclidean':
		pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
	elif distance == 'cosine':
		pairwise_distance_function = partial(pairwise_cosine, device=device)
	elif distance == 'EMD':
		pairwise_distance_function = partial(pairwise_EMD, device=device)
	else:
		raise NotImplementedError

	# convert to float
	if type(X) != torch.Tensor:
		X = torch.tensor(X)
	X = X.float()

	# transfer to device
	X = X.to(device)

	dis = pairwise_distance_function(X, centroids)
	if (len(dis.shape) == 1): # Prediction on a single data
		choice_cluster = torch.argmin(dis)
		
	else: 
		choice_cluster = torch.argmin(dis, dim=1)

	return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), tqdm_flag=True):
	if tqdm_flag:
		print(f'device is :{device}')
	
	# transfer to device
	data1, data2 = data1.to(device), data2.to(device)

	A = data1.unsqueeze(dim=1) # N*1*M
	B = data2.unsqueeze(dim=0) # 1*N*M

	dis = (A - B) ** 2.0
	# return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis


def pairwise_cosine(data1, data2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	# transfer to device
	data1, data2 = data1.to(device), data2.to(device)

	A = data1.unsqueeze(dim=1) # N*1*M
	B = data2.unsqueeze(dim=0) # 1*N*M

	# normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
	A_normalized = A / A.norm(dim=-1, keepdim=True)
	B_normalized = B / B.norm(dim=-1, keepdim=True)

	cosine = A_normalized * B_normalized

	# return N*N matrix for pairwise distance
	cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
	return cosine_dis


def pairwise_EMD(data1, data2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	assert(len(data1.shape) == 2)
	assert(len(data2.shape) == 2)
	assert(data1.shape[1] == data2.shape[1])
	n = data1.shape[1]
	pos_a = torch.tensor([[i] for i in range(n)])
	pos_b = torch.tensor([[i] for i in range(n)])
	C = ot.dist(pos_a, pos_b, metric='euclidean')
	C.to(device)

	# Correct solution, but very slow
	dist = torch.zeros((data1.shape[0], data2.shape[0]))
	for i, hist_a in enumerate(data1):
		for j, hist_b in enumerate(data2):
			for _ in range(10):  # Janky fix for small precision error
				try:
					ot_emd = ot.emd(hist_a, hist_b, C, numThreads="max")  # The precision is set to 7, so sometimes the sum doesn't get to precisely 1. 
					break
				except Exception as e:
					print(e)
					continue
				
			transport_cost_matrix = ot_emd * C
			dist[i][j] = transport_cost_matrix.sum()
	
	return dist

def kmeans_search(X):
	"""
	We can check for the quality of clustering by checking the inter-cluster distance, using the 
	same metric that we used for EMD.
	
	At some point, there is no point in increasing the number of clusters, since we don't really
	get more information.
	"""
	# Search for the optimal number of clusters through a grid like search

	if type(X) != torch.Tensor:
		X = torch.tensor(X)
	# convert to float
	X = X.float()

	# n_clusters = [10, 25, 50, 100, 200, 1000, 5000]
	n_clusters = [5000]
	for n_cluster in n_clusters:
		cluster_indices, centroids = kmeans(X, n_cluster)
		X_cluster_centroids = centroids[cluster_indices]
		distances = 0
		for i, X_cluster_centroid in enumerate(X_cluster_centroids):
			distances += pairwise_distance(torch.unsqueeze(X_cluster_centroid, axis=0), torch.unsqueeze(X[i], axis=0), tqdm_flag=False)
		print(f"Sum of cluster to data distance {distances}")
		print(f"Mean cluster to data distance {distances / X_cluster_centroids.shape[0]}")


	


