import time
import numpy as np
from mpi4py import MPI
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X, y = make_classification(n_samples=100000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

params = [{'mlp_layer1': [16, 32],
           'mlp_layer2': [16, 32], 
           'mlp_layer3': [16, 32]}]

pg = list(ParameterGrid(params))
    
if rank == 0:
    start_time = time.time()

    # o master xwrizei ta tasks kai ta stelnei se kathe worker
    worker_tasks = np.array_split(pg, size - 1)

    for i, tasks in enumerate(worker_tasks):
        comm.send(tasks, dest=i+1, tag=0)

    # dexetai ta work(tasks) apo olous toys workers
    results = []
    for i in range(1, size):
        worker_results = comm.recv(source=i, tag=0)
        results.extend(worker_results)

    for r in results:
        print(r)
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")    

else:

    def work(p):
        l1 = p['mlp_layer1']
        l2 = p['mlp_layer2']
        l3 = p['mlp_layer3']
        m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        ac = accuracy_score(y_pred, y_test)
        return (rank, p, ac)
    
    tasks = comm.recv(source=0, tag=0)
    worker_results = list(map(work, tasks))
    comm.send(worker_results, dest=0, tag=0)
