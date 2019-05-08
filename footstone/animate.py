# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:03:03 2019

@author: Red
Refer to https://zhuanlan.zhihu.com/p/36727011

from moviepy.editor import VideoClip

def make_frame(t):
    # returns an image of the frame at time t
    # ... 用任意库创建帧
    return frame_for_time_t # (Height x Width x 3) Numpy array

animation = VideoClip(make_frame, duration=3) # 3-second clip

# 支持导出为多种格式
animation.write_videofile("my_animation.mp4", fps=24) # 导出为视频
animation.write_gif("my_animation.gif", fps=24) # 导出为GIF
"""
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

def play_svm():
    from sklearn import svm # sklearn = scikit-learn
    from sklearn.datasets import make_moons

    X, Y = make_moons(50, noise=0.1, random_state=2) # 半随机数据
    
    fig, ax = plt.subplots(1, figsize=(4, 4), facecolor=(1,1,1))
    fig.subplots_adjust(left=0, right=1, bottom=0)
    xx, yy = np.meshgrid(np.linspace(-2,3,500), np.linspace(-1,2,500))
    
    def make_frame(t):
        ax.clear()
        ax.axis('off')
        ax.set_title("SVC classification", fontsize=16)
    
        classifier = svm.SVC(gamma=2, C=1)
        
        # 不断变化的权重让数据点一个接一个的出现
        weights = np.minimum(1, np.maximum(0, t**2+10-np.arange(50)))
        classifier.fit(X, Y, sample_weight=weights)
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.8,
                    vmin=-2.5, vmax=2.5, levels=np.linspace(-2,2,20))
        ax.scatter(X[:,0], X[:,1], c=Y, s=50*weights, cmap=plt.cm.bone)
    
        return mplfig_to_npimage(fig)
    
    animation = VideoClip(make_frame, duration = 7)
    animation.write_gif("svm.gif", fps=15)
 
def play_numpy():
    import numpy as np
    from scipy.ndimage.filters import convolve
    import moviepy.editor as mpy
    
    file = r"imgs\france_density.png"
    
    #### 参数和约束条件
    infection_rate = 0.3
    incubation_rate = 0.1
    
    dispersion_rates  = [0, 0.07, 0.03] # for S, I, R
    
    # 该内核会模拟人类/僵尸如何用一个位置扩散至邻近位置
    dispersion_kernel = np.array([[0.5, 1 , 0.5],
                                    [1  , -6, 1],
                                    [0.5, 1, 0.5]]) 
    
    france = mpy.ImageClip(file).resize(width=400)
    SIR = np.zeros( (3,france.h, france.w),  dtype=float)
    SIR[0] = france.get_frame(0).mean(axis=2)/255
    
    start = int(0.6*france.h), int(0.737*france.w)
    SIR[1,start[0], start[1]] = 0.8 # infection in Grenoble at t=0
    
    dt = 1.0 # 一次更新=实时1个小时
    hours_per_second= 7*24 # one second in the video = one week in the model
    world = {'SIR':SIR, 't':0}
    
    ##### 建模
    def infection(SIR, infection_rate, incubation_rate):
        """ Computes the evolution of #Sane, #Infected, #Rampaging"""
        S,I,R = SIR
        newly_infected = infection_rate*R*S
        newly_rampaging = incubation_rate*I
        dS = - newly_infected
        dI = newly_infected - newly_rampaging
        dR = newly_rampaging
        return np.array([dS, dI, dR])
    
    def dispersion(SIR, dispersion_kernel, dispersion_rates):
        """ Computes the dispersion (spread) of people """
        return np.array( [convolve(e, dispersion_kernel, cval=0)*r
                           for (e,r) in zip(SIR, dispersion_rates)])
    
    def update(world):
        """ spread the epidemic for one time step """
        infect = infection(world['SIR'], infection_rate, incubation_rate)
        disperse = dispersion(world['SIR'], dispersion_kernel, dispersion_rates)
        world['SIR'] += dt*( infect + disperse)
        world['t'] += dt
    
    # 用MoviePy制作动画
    def world_to_npimage(world):
        """ Converts the world's map into a RGB image for the final video."""
        coefs = np.array([2,25,25]).reshape((3,1,1))
        accentuated_world = 255*coefs*world['SIR']
        image = accentuated_world[::-1].swapaxes(0,2).swapaxes(0,1)
        return np.minimum(255, image)
    
    def make_frame(t):
        """ Return the frame for time t """
        while world['t'] < hours_per_second*t:
            update(world)
        return world_to_npimage(world)
 
    animation = mpy.VideoClip(make_frame, duration=25)
    
    # 可以将结果写为视频或GIF（速度较慢）
    #animation.write_gif(make_frame, fps=15)
    animation.write_videofile(r'out\infect.mp4', fps=20)    

import mayavi.mlab as mlab
mlab.figure(size = (500, 500),\
            bgcolor = (1,1,1), fgcolor = (0.5, 0.5, 0.5))
f = mlab.gcf()
f.scene._lift()
count = 0
t = np.linspace(0, 5*np.pi, 100)
#mlab.view(azimuth=0, elevation=90)
def myframe(time):
    global count
    """ Demo the bar chart plot with a 2D array.
    """
    print("generate barchar:", time)

    mlab.clf()
    tmp = t[:count+1]
    mlab.plot3d(np.sin(tmp), np.cos(tmp), 0.1*tmp, tmp)
    count += 1
    return mlab.screenshot(antialiased=False)

if __name__ == "__main__":
    import moviepy.editor as mpy
    animation = mpy.VideoClip(myframe, duration=1)
    
    # 可以将结果写为视频或GIF（速度较慢）
    animation.write_gif(r"out\test.gif", fps=10, verbose=False)
    #animation.write_videofile(r'out\test.mp4', fps=10, verbose=False)    
