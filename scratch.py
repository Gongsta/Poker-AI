import torch
import time
import ot

	
if __name__ == "__main__":
	start = time.time()	
	a = torch.tensor([[1.0,0,0]] * 10000)
	b = torch.tensor([[0,0,1.0]] * 100)
	print(pairwise_EMD(a,b))
	print(time.time() - start)
