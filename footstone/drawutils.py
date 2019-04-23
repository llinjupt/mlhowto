import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

'''
(a) Ax + By + C = 0
(b) (y - y0) / (x - x0) = B / A

=>
x = (B*B*x0 - A*B*y0 - A*C) / (A*A + B*B)
y = (-A*B*x0 + A*A*y0 - B*C) / (A*A + B*B)
'''
def get_footpoint(px, py, w_):
    if w_.shape[0] != 3:
        print("can't calculate footpoint with {}".formate(w_))
        return None
    
    A,B,C = w_[1], w_[2], w_[0]
    x = (B*B*px - A*B*py - A*C)/(A*A + B*B)
    y = (-A*B*px + A*A*py - B*C)/(A*A + B*B)
    
    return x, y

def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def rotation_transform(theta):
    ''' rotation matrix given theta
    Inputs:
        theta    - theta (in degrees)
    '''
    theta = np.radians(theta)
    A = [[np.math.cos(theta), -np.math.sin(theta)],
         [np.math.sin(theta), np.math.cos(theta)]]
    return np.array(A)

# resolution is step size in the mesh
def plot_decision_regions(X, y, clf, test_idx=None, resolution=0.01):
    from matplotlib.colors import ListedColormap
    # setup marker generator and color map
    markers = ('s', 'x', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    
    plt.title("Decision surface of multi-class")
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, s=50,
                    edgecolor='black')

    if test_idx is None:
        return

    # plot all samples with cycles
    X_test = X[test_idx, :]
    plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black',
                alpha=1.0, linewidth=1, marker='o', s=50, 
                label='test dataset')

def test_plot_decision_regions():
    import dbload
    from sklearn.linear_model import Perceptron
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = dbload.load_iris_mclass()
    ppn = Perceptron(max_iter=100, eta0=0.01, random_state=1)
    ppn.fit(X_train, y_train)
    predict = ppn.predict(X_test)
    print("Misclassified number {}, Accuracy {:.2f}%".format((predict != y_test).sum(), 
           accuracy_score(y_test, predict)*100))

    X_all = np.vstack((X_train, X_test))
    y_all = np.hstack((y_train, y_test))
    print(y_all[0:20])
    plot_decision_regions(X_all, y_all, clf=ppn, 
                          test_idx=range(X_train.shape[0], X_all.shape[0]))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def relu(z):
    np.clip(z, 0, np.finfo(z.dtype).max, out=z)
    return z

def tanh(X):
    return np.tanh(X, out=X)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def plot_sigmoid():
    z = np.arange(-8, 8.1, 0.1)
    y = sigmoid(z)
    
    plt.plot(z, y)
    plt.title("Sigmoid Function")
    plt.axvline(0.0, color='k')
    plt.ylim(0, 1)
    plt.xlim(np.min(z), np.max(z))
    plt.xlabel('z')
    plt.ylabel('sigmoid(z)')
    
    # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    plt.plot((-8, 8), (0.5, 0.5), lw=1, c='gray')
    
    plt.tight_layout()
    plt.show()

def plot_sigmoid_with_odd():
    z = np.arange(-8, 8.1, 0.1)
    y = sigmoid(z)
    
    plt.title("Sigmoid Function")
    plt.plot(z, y)
    
    xm = 2 # plot vertical line to show p and 1-p relationship
    plt.plot([xm,xm],[0,sigmoid(xm)], color ='blue', linewidth=1.5, linestyle="--")
    plt.text(xm+0.1, (sigmoid(xm) / 2), r'$p=sigmoid(z)$')
    
    plt.plot([xm,xm],[sigmoid(xm),1], color ='orange', linewidth=1.5, linestyle="--")
    plt.text(xm+0.1, ((1 + sigmoid(xm)) / 2), r'$1-p$')
    
    xm = -2
    plt.plot([xm,xm],[0,sigmoid(xm)], color ='orange', linewidth=1.5, linestyle="--")
    plt.text(xm+0.1, (sigmoid(xm) / 2), r'$1-p$')
    
    plt.plot([xm,xm],[sigmoid(xm),1], color ='blue', linewidth=1.5, linestyle="--")
    plt.text(xm-4, ((1 + sigmoid(xm)) / 2), r'$p=1-sigmoid(z)$')
    
    plt.axvline(0.0, color='k')
    plt.ylim(0, 1)
    plt.xlim(np.min(z), np.max(z))
    plt.xlabel('z')
    plt.ylabel('sigmoid(z)')
    
    # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    plt.plot((-8, 8), (0.5, 0.5), lw=1, c='gray')
    
    plt.tight_layout()
    plt.show()

def plot_sigmoid_cost():
    def cost_1(z):
        return - np.log(sigmoid(z))

    def cost_0(z):
        return - np.log(1 - sigmoid(z))

    z = np.arange(-10, 10, 0.1)
    phi_z = sigmoid(z)
    
    c1 = [cost_1(x) for x in z]
    plt.plot(phi_z, c1, label='j(w) if y=1', c='blue')
    
    c0 = [cost_0(x) for x in z]
    plt.plot(phi_z, c0, linestyle='--', label='j(w) if y=0', c='orange')
    
    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel('$\phi$(z)')
    plt.ylabel('j(w)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_lines(funcs):
    colors = ['black', 'gray', 'red', 'blue', 'cyan', 'purple']
    colors = colors * (len(funcs) // len(colors) + 1)
    
    num = len(funcs)
    x = np.linspace(-2, 2, num=50)
    plt.figure(figsize=(6,4))
    for i, f, c in zip(range(num), funcs, colors):
        c = str((1 - i/(num + 1)) * 0.80)
        lab = '$y=x+' + str(0.1 * (i + 1))[0:3] + '*x^5$' 
        plt.plot(x, f(x), c, label=lab)
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

def plot_funcs():
    from functools import partial
    def tmppower(x, n=1,delta=1.0):
        return np.power(x, 1) + np.power(x, n) * delta
    funcs = [partial(tmppower, n=5, delta=i*0.1) for i in range(6)]
    plot_lines(funcs)

def plot_showimgs(imgs, title=(), tight=True):
    plt.figure(figsize=(8,8))
    plt.title(title)
    
    count = len(imgs)
    columns = rows = int(count ** 0.5)
    if columns ** 2 < count:
        columns += 1

    if columns * rows < count:
        rows += 1
    
    index = 1
    for i in imgs:
        plt.subplot(rows, columns, index)
        plt.xticks([])
        plt.yticks([])
        
        if len(title) >= index:
            plt.title(title[index - 1])
        plt.imshow(i, cmap='gray', interpolation='none') 
        plt.axis('off')
        index += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    if tight: plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass
