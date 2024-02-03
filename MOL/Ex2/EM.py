import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import sqrt, pi
from matplotlib.widgets import Slider, Button
np.random.seed(3)

colors = ["#cccc00", "#006600", "m", "b", "r"]

def EM(data, K, n_iter = 25, return_all_iterations=False):
    N, M = data.shape # assume: N is number of data points, M is each data point's dimension

    sigma_def = 3

    # random initialization of cluster centers
    m = np.zeros((K,M))
    for mm in range(M):
        m[:, mm] = np.random.uniform(np.min(data[:,mm]), np.max(data[:,mm]), (K, ))
    Sigma = [sigma_def*np.eye(M) for kk in range(K)]
    gamma = 1/K*np.ones((N,K), dtype=np.int32)
    p = 1/K*np.ones((K,))
    gamma_old = None
    
      
    # if we want to get back all parameters for each single iteration    
    if return_all_iterations:
        Sigma_history = [[np.copy(Sigma[k]) for k in range(K)]]
        m_history = np.zeros((1, K, M))
        m_history[0, :, :] = m
        gamma_history = np.zeros((1, N, K))
        gamma_history[0, :, :] = gamma
        p_history = np.zeros((1,K))
        p_history[0, :] = p
    # if we are only interested in the final clustering result        
    else:
        m_history = None
        Sigma_history = None
        gamma_history = None
        p_history = None
    
    
    for it in range(n_iter):
        # assignment step
        for n in range(N):
            x = data[n, :]
            terms = np.array([p[k]*1/sqrt((2*pi)**M*np.linalg.det(Sigma[k])) * np.exp(-0.5*(x-m[k,:]) @ (np.linalg.inv(Sigma[k]) @ (x-m[k,:]))) for k in range(K)])
            gamma[n, :] = terms/np.sum(terms)
        
        
            
        if return_all_iterations:
            gamma_history = np.append(gamma_history, [gamma], 0)
        

        # update step
        means = np.dot(gamma.T, data)
        Nk = [np.sum(gamma[:, k]) for k in range(K)]
        m_new = np.array([ means[k, :]/Nk[k] for k in range(K)])
        
        # if we haven't moved a lot, break the loop and stick with current clustering
        if np.sum(np.sum((m-m_new)*(m-m_new))) < 10**-8:
            break
        
        m = m_new
        
        Sigma_new = [np.zeros((M, M)) for k in range(K)]
        for k in range(K):
            Sigmak = np.zeros((M,M))
            for n in range(N):
                x = data[n, :]
                Sigmak += gamma[n, k] * np.dot((x-m[k,:]).reshape((-1,1)), (x-m[k,:]).reshape((1,-1)))
            Sigma_new[k] = Sigmak/Nk[k]
            if np.linalg.det(Sigma_new[k]) < 1e-10:
                print("Covariance singularity!")
                return {"success": False, "m": None, "Sigma": None, "p": None, "gamma": None, "m_history": None, "Sigma_history": None, "p_history": None, "gamma_history": None}
        
        Sigma = [np.copy(Sigma_new[k]) for k in range(K)]

        p = np.array([Nk[k]/N for k in range(K)])

        
        if return_all_iterations:
            m_history = np.append(m_history, [m], 0)
            Sigma_history.append([np.copy(Sigma_new[k]) for k in range(K)])
            p_history = np.append(p_history, [p], 0)
        
        
    return {"success": True, "m": m, "Sigma": Sigma, "p": p, "gamma": gamma, "m_history": m_history, "Sigma_history": Sigma_history, "p_history": p_history, "gamma_history": gamma_history} # return a dictionary 


# matplotlib ellipse plotter (from matplotlib.org)


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def plot_ellipse(mean, cov, ax, n_std=1.0, color='k', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    #cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        color=color, alpha = 0.05,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])
    

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    # Test cases. Fiddle with parameters:
    n_iter = 100
    K = 2 # number of clusters
    
    
    option = 8  # select scenario 
    
    # description of options:
    # option 1: three clusters with few points
    # option 2: two elongated clusters
    # option 3: one large, one small cluster
    # option 4: small, dense cluster within large cluster
    # option 5: two ring-shaped clusters
    # option 6: faithful.csv
    # option 7: independent.csv
    # option 8: faithful.csv, K=3
    
    

    # END parameters
    
    if option == 1:
        mean1 = np.array([1,0])
        sig1 = 0.05*np.eye(2)
        mean2 = np.array([0, 1])
        sig2 = 0.05*np.eye(2)
        mean3 = np.array([2, 1])
        sig3 = 0.05*np.eye(2)
        data = np.concatenate((np.random.multivariate_normal(mean1, sig1, 5), np.random.multivariate_normal(mean2, sig2, 5), np.random.multivariate_normal(mean3, sig3, 5)))
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
        mean2 = np.array([3.5, 1])
        sig2 = np.diag([0.1, 0.1])
        data = np.concatenate((np.random.multivariate_normal(mean1, sig1, 50), np.random.multivariate_normal(mean2, sig2, 50)))
    elif option == 5:
        data1 = np.random.normal(0, 1, (40,2))
        norms1 = np.tile(np.sqrt(data1[:,0]**2 + data1[:, 1]**2), (2, 1)).T
        data1 = data1/(norms1) + np.random.normal(0, 0.05, (40,2))        
        data2 = np.random.normal(0, 1, (40,2))
        norms2 = np.tile(np.sqrt(data2[:,0]**2 + data2[:, 1]**2), (2, 1)).T
        data2 = data2/(0.3*norms2) + np.random.normal(0, 0.05, (40,2))
        data = np.concatenate((data1, data2))
    elif option == 6:
        import pandas as pd
        df = pd.read_csv("faithful.csv")
        print(df) # see what it looks like
        data = df.loc[:,["eruptions", "waiting"]].to_numpy() 
        #ret = KMeans(data, K, return_all_iterations=show_iterations)
    elif option == 7:
        import pandas as pd
        df = pd.read_csv("independent.csv")
        print(df) # see what it looks like
        data = df.loc[:,["eruptions", "waiting"]].to_numpy() 
        #ret = KMeans(data, K, return_all_iterations=show_iterations)
    elif option == 8:
        import pandas as pd
        K=3
        df = pd.read_csv("faithful.csv")
        print(df) # see what it looks like
        data = df.loc[:,["eruptions", "waiting"]].to_numpy() 
        #ret = KMeans(data, K, return_all_iterations=show_iterations)
    
    # find bounding box of data (for plotting purposes)
    xmin = np.min(data[:, 0])
    xmax = np.max(data[:, 0])
    ymin = np.min(data[:, 1])
    ymax = np.max(data[:, 1])
    dmin = min(xmin, ymin)
    dmax = max(xmax, ymax)
    
    #data = np.concatenate((np.random.normal(0, 0.05, (10,)), np.random.normal(1, 0.05, (10,)))).reshape((-1,1))
    
    
    N, M = data.shape
    
    ret = EM(data, K, n_iter = n_iter, return_all_iterations=True)
    
    
    m = ret["m"]
    m_history = ret["m_history"]
    
    n_iter = m_history.shape[0] - 1
    
    Sigma = ret["Sigma"]
    Sigma_history = ret["Sigma_history"]
    p = ret["p"]
    p_history = ret["p_history"]
    gamma = ret["gamma"]
    gamma_history = ret["gamma_history"]
    
    plt.ion()
    
    
    
    def drawfig(it):
        plt.figure()
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        ax1.set_xlim([xmin-0.5, xmax+0.5])
        ax1.set_ylim([ymin-0.5, ymax+0.5])
        m_now = m_history[it+1, :, :]
        m_old = m_history[it, :, :]
        
        gamma_now = gamma_history[it+1, :]
        Sigma_now = Sigma_history[it+1]
        Sigma_old = Sigma_history[it]

        
        for k in range(K):
            ax1.plot(m_old[k, 0], m_old[k, 1], color = colors[k], marker = "s", markeredgecolor="k", markersize=10)
        
        
        
        if K == 2: # linear interpolation between colors possible
            mixcoeffs = gamma_now[:, 1]
            sc = ax1.scatter(data[:, 0], data[:, 1], c=[colorFader(colors[0],colors[1],mixc) for mixc in mixcoeffs], s=45, zorder=2)
            
        else: # for more clusters we can only color with the dominant cluster :(
            cs_index = [int(list(gamma_now[n, :]).index(max(gamma_now[n, :]))) for n in range(N)]
            ax1.scatter(data[:, 0], data[:, 1], c=[colors[csi] for csi in cs_index], s=45, zorder=2)
            
        for k in range(K):
            plot_ellipse(m_old[k, :], Sigma_old[k], ax1)
            plot_ellipse(m_old[k, :], Sigma_old[k], ax1, n_std=2.0)
        

        
        
        ax2.set_xlim([xmin-0.5, xmax+0.5])
        ax2.set_ylim([ymin-0.5, ymax+0.5])
        
        for k in range(K):
            ax2.plot(m_now[k, 0], m_now[k, 1], color = colors[k], marker = "s", markeredgecolor="k", markersize=10)
                        
                        
                        
                            
        if K == 2: # linear interpolation between colors possible
            mixcoeffs = gamma_now[:, 1]
            sc = ax2.scatter(data[:, 0], data[:, 1], c=[colorFader(colors[0],colors[1],mixc) for mixc in mixcoeffs], s=45, zorder=2)
            
        else: # for more clusters we can only color with the dominant cluster :(
            cs_index = [int(list(gamma_now[n, :]).index(max(gamma_now[n, :]))) for n in range(N)]
            ax2.scatter(data[:, 0], data[:, 1], c=[colors[csi] for csi in cs_index], s=45, zorder=2)
        for k in range(K):
            plot_ellipse(m_now[k, :], Sigma_now[k], ax2)
            plot_ellipse(m_now[k, :], Sigma_now[k], ax2, n_std=2.0)
        
        ax1.set_title("iteration " + str(it) + ": after assignment")
        ax2.set_title("iteration " + str(it) + ": after update")
        #plt.title("iteration " + str(it) + ": after update")
        # ax1.set_aspect('equal')
        # ax2.set_aspect('equal')
        plt.show()
        plt.savefig(f"figure_{it}.png")
    
    for i in range(m_history.shape[0]-1):
        drawfig(i)
        #drawfig(m_history.shape[0]-2)
    
  

    
