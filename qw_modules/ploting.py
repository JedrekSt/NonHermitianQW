from matplotlib import pyplot as plt
import numpy as np

def berry_bloch_phase(th1,th2,ns,th_1,th_2,data_ans):

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2)

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax1.plot_surface(x, y, z, color='pink', alpha=0.1, rstride=10, cstride=10, edgecolor='black')

    vectors = [
        [1, 0, 0],  
        [0, 1, 0], 
        [0, 0, 1],  
    ]

    origin = [0, 0, 0]  
    for v in vectors:
        ax1.quiver(*origin, *v, color='g', linewidth=1,  normalize=True)

    for i in range(ns[0].shape[1]):
        ax1.quiver(*origin, *ns[1][:,i], color='b', linewidth=1,alpha = 0.3)
    for i in range(ns[0].shape[1]):
        ax1.quiver(*origin, *ns[0][:,i], color='r', linewidth=1,alpha = 0.3)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Bloch vectors for k in BZ')
    ax1.set_box_aspect([1, 1, 1])
    ax1.grid(False)

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])

    cax = ax2.contourf(th_1,th_2,data_ans/np.pi,cmap = "viridis")
    fig.colorbar(cax)
    ax2.set_xlabel(r"$\theta_{2}$")
    ax2.set_ylabel(r"$\theta_{1}$")
    ax2.set_title("winding number for single step QW")
    ax2.scatter([th1],[th2],color="blue", alpha = 1)


    return ax1,ax2
