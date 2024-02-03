import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

np.random.seed(12)

colors = ["#cccc00", "#006600", "m", "b", "r"]

def KMeans(data, K, return_all_iterations=False):
    N, M = data.shape # assume: N is number of data points, M is each data point's dimension
    
    # random initialization of cluster centers
    m = np.random.uniform(np.min(data), np.max(data), (K, M))
    a = np.zeros((N,), dtype=np.int32)
    a_old = None
    
    if return_all_iterations:
        a_history = np.zeros((1, N), dtype=np.int32)
        a_history[0, :] = a
        m_history = np.zeros((1, K, M))
        m_history[0, :] = m
    else:
        a_history = None
        m_history = None
    
    it = 1
    while True:
        # assignment step
        for n in range(N):
            x = data[n, :]
            distances = [np.dot(x-m[k, :], x-m[k, :]) for k in range(K)]
            a[n] = int(distances.index(min(distances)))
        
        # if assignment did not change anything, break loop and return value afterwards
        if a_old is not None and np.dot(a-a_old, a-a_old) < 1e-10:
            break
            
        if return_all_iterations:
            a_history = np.append(a_history, [a], 0)
        

        # update step
        
        means = np.zeros((K, M))
        numelems = np.zeros((K, ))
        for n in range(N):
            means[a[n], :] += data[n, :]
            numelems[a[n]] += 1
            
        for k in range(K):
            if numelems[k] == 0:
                continue
            m[k, :] = means[k, :]/numelems[k]
        
        if return_all_iterations:
            m_history = np.append(m_history, [m], 0)
        
        # copy values of a into a_old. Use this if you need to copy numpy arrays, 
        # instead of just setting a_old = a (side effects possible because this creates a reference)
        a_old = np.copy(a)
        
        it += 1
        
    return {"m": m, "a": a, "m_history": m_history, "a_history": a_history} # return a dictionary 



if __name__ == "__main__":
    # Test cases. Fiddle with parameters:
    
    K = 2 # number of clusters
    show_iterations = True # False if you only want to see result of K-means
    option = 5 # select scenario (1, 2, 3, 4)

    # END parameters
    
    if option == 1:
        mean1 = np.array([1,0])
        sig1 = 0.1*np.eye(2)
        mean2 = np.array([0, 1])
        sig2 = 0.1*np.eye(2)
        data = np.concatenate((np.random.multivariate_normal(mean1, sig1, 5), np.random.multivariate_normal(mean2, sig2, 5)))
    elif option == 2:
        mean1 = np.array([1,0])
        sig1 = np.diag([1.5, 0.05])
        mean2 = np.array([1, 2])
        sig2 = np.diag([1.5, 0.05])
        data = np.concatenate((np.random.multivariate_normal(mean1, sig1, 25), np.random.multivariate_normal(mean2, sig2, 25)))
    elif option == 3:
        mean1 = np.array([4, 5])
        sig1 = np.diag([0.5, 3])
        mean2 = np.array([7, 5])
        sig2 = np.diag([0.2, 0.2])
        data = np.concatenate((np.random.multivariate_normal(mean1, sig1, 50), np.random.multivariate_normal(mean2, sig2, 10)))    
    elif option == 4:
        mean1 = np.array([1, 1])
        sig1 = np.diag([4, 4])
        mean2 = np.array([3, 1])
        sig2 = np.diag([0.1, 0.1])
        data = np.concatenate((np.random.multivariate_normal(mean1, sig1, 50), np.random.multivariate_normal(mean2, sig2, 50)))
    elif option == 5:
        import pandas as pd
        df = pd.read_csv("faithful.csv")
        print(df) # see what it looks like
        data = df.loc[:,["eruptions", "waiting"]].to_numpy() 
    
    # find bounding box of data (for plotting purposes)
    xmin = np.min(data[:, 0])
    xmax = np.max(data[:, 0])
    ymin = np.min(data[:, 1])
    ymax = np.max(data[:, 1])
    
    dmin = min(xmin, ymin)
    dmax = max(xmax, ymax)
    
    N, M = data.shape
    
    ret = KMeans(data, K, return_all_iterations=show_iterations)
    m = ret["m"]
    m_history = ret["m_history"]
    a = ret["a"]
    a_history = ret["a_history"]
    
    
    if M == 2:
        if m_history is not None:
            num_its = m_history.shape[0]
            fig = plt.figure(figsize=(8,4))
            ax1 = plt.subplot(1,2,1)
            plt.xlim([xmin-0.5, xmax+0.5])
            plt.ylim([ymin-0.5, ymax+0.5])
            
            for k in range(K):
                plt.plot(m_history[0, k, 0], m_history[0, k, 1], color = colors[k], marker = "s", markeredgecolor="k", markersize=10)
            
            plt.scatter(data[:, 0], data[:, 1], color="k", s=45)
            #plt.axis("equal")
            #plt.title("n = " + str(it) + ", after update")
            plt.title("initial state")
            plt.tight_layout()
            ax2 = plt.subplot(1,2,2)
            if K >= 3:
                vor = Voronoi(m)
                voronoi_plot_2d(vor, show_vertices=False, ax=ax2)
            plt.xlim([xmin-0.5, xmax+0.5])
            plt.ylim([ymin-0.5, ymax+0.5])
            
            for k in range(K):
                plt.plot(m_history[-1, k, 0], m_history[-1, k, 1], color = colors[k], marker = "s", markeredgecolor="k", markersize=10)
            
            
            
            # compute and plot separator line (for K = 2)
            if K == 2:
                v = m[1, :] - m[0, :]
                a0 = np.dot(v, m[0, :])
                a1 = np.dot(v, m[1, :])
                slope = -v[0]/v[1]
                intercept = (m[0, 1] + m[1, 1])/2 - slope*(m[0, 0] + m[1, 0])/2
                sep = lambda x: slope*x + intercept
                plt.plot([xmin, xmax], [sep(xmin), sep(xmax)], '--')
            
            plt.scatter(data[:, 0], data[:, 1], color=[colors[aa] for aa in a], s=45)
            #plt.axis("equal")
            plt.xlim([xmin-0.5, xmax+0.5])
            plt.ylim([ymin-0.5, ymax+0.5])
            plt.title("final state")
            plt.tight_layout()
            # full visualization of all iterations
            for it in range(num_its-1):
                m_now = m_history[it+1, :, :]
                m_old = m_history[it, :, :]
                
                a_now = a_history[it+1, :]
                
                plt.figure(figsize=(8,4))
                plt.ion()
                ax1 = plt.subplot(1,2,1)
                if K >= 3:
                    vor = Voronoi(m_old)
                    voronoi_plot_2d(vor, show_vertices=False, ax=ax1)
                plt.xlim([xmin-0.5, xmax+0.5])
                plt.ylim([ymin-0.5, ymax+0.5])
                for k in range(K):
                    plt.plot(m_old[k, 0], m_old[k, 1], color = colors[k], marker = "s", markeredgecolor="k", markersize=10)
                
                # compute and plot separator line (for K = 2)
                if K == 2:
                    v = m_old[1, :] - m_old[0, :]
                    a0 = np.dot(v, m_old[0, :])
                    a1 = np.dot(v, m_old[1, :])
                    slope = -v[0]/v[1]
                    intercept = (m_old[0, 1] + m_old[1, 1])/2 - slope*(m_old[0, 0] + m_old[1, 0])/2
                    sep = lambda x: slope*x + intercept
                    plt.plot([xmin, xmax], [sep(xmin), sep(xmax)], '--')
                    
                plt.scatter(data[:, 0], data[:, 1], c=[colors[aa] for aa in a_now], s=45, zorder=2)
                plt.title("n = " + str(it) + ", after assignment")
                
                
                ax2 = plt.subplot(1,2,2)
                plt.xlim([xmin-0.5, xmax+0.5])
                plt.ylim([ymin-0.5, ymax+0.5])
                
                for k in range(K):
                    plt.plot(m_old[k, 0], m_old[k, 1], color = colors[k], marker = "x")
                    plt.plot(m_now[k, 0], m_now[k, 1], color = colors[k], marker = "s", markeredgecolor="k", markersize=10)
                    plt.plot([m_now[k, 0], m_old[k, 0]], [m_now[k, 1], m_old[k, 1]], color = colors[k], linestyle = "--") 
                
                plt.scatter(data[:, 0], data[:, 1], color=[colors[aa] for aa in a_now], s=45)
                plt.title("n = " + str(it) + ", after update")
                plt.tight_layout()
                plt.show()
                plt.savefig(f"fig_{it}.png")
        
        else:
            fig = plt.figure()
            plt.ion()
            plt.xlim([xmin-0.5, xmax+0.5])
            plt.ylim([ymin-0.5, ymax+0.5])
            if K >= 3:
                vor = Voronoi(m)
                voronoi_plot_2d(vor, show_vertices=False, ax=fig.gca())
            for k in range(K):
                plt.plot(m[k, 0], m[k, 1], color = colors[k], marker = "s", markeredgecolor="k", markersize=10)
            
            # compute and plot separator line (for K = 2)
            if K == 2:
                v = m[1, :] - m[0, :]
                a0 = np.dot(v, m[0, :])
                a1 = np.dot(v, m[1, :])
                slope = -v[0]/v[1]
                intercept = (m[0, 1] + m[1, 1])/2 - slope*(m[0, 0] + m[1, 0])/2
                sep = lambda x: slope*x + intercept
                plt.plot([xmin, xmax], [sep(xmin), sep(xmax)], '--')
            
            plt.scatter(data[:, 0], data[:, 1], color=[colors[aa] for aa in a], s=45)
            #plt.axis("equal")
            plt.xlim([xmin-0.5, xmax+0.5])
            plt.ylim([ymin-0.5, ymax+0.5])
                
            plt.show()
            plt.savefig("figfinal.png")
    else: 
        print("Visualization only in 2d")
    

    

    
