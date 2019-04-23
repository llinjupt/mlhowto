# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:10:34 2018
pltnn: plot neuron net with simple nodes and layers 

@author: Red lli_njupt@163.com MIT License
"""

from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def relu(z):
    np.clip(z, 0, np.finfo(z.dtype).max, out=z)
    return z

def tanh(X):
    return np.tanh(X, out=X)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class pltNN():
    # type support 'logistic', 'relu' and 'tanh'
    def __init__(self, x_range=5, w_range=10, type='logistic'):
        # x axis scope [-x_range, x_range]
        self.x1 = np.arange(-x_range, x_range, 0.001)
        self.x_range = x_range
        self.w_range = w_range
        self.acttype = type
        
        # reset value
        self.w_reset = 0

    def activate(self, z):
        if self.acttype == 'tanh':
            return tanh(z)
        elif self.acttype == 'relu':
            return relu(z)
        
        return sigmoid(z)

    def activate11(self, w1, b):
        z = w1 * self.x1 + b
        return self.activate(z)

    #   w1*x + w0
    # x----------->y
    def plot_neuron_11(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        plt.title('$' + self.acttype + '(w_1*x+w_0)$')
        # 使用默认值绘制图像
        s = self.activate11(self.w_reset, self.w_reset)
        l, = plt.plot(self.x1, s, lw=2, color='red') 
        # x 轴范围和 y 轴范围
        plt.axis([-self.x_range, self.x_range, 0, 1])
        
        # 滑动条背景颜色
        axcolor = 'white'
        w0_bar = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        w1_bar = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        
        # 绑定滑动条和 reset 数值
        w0 = Slider(w0_bar, 'w0', -self.w_range*3, self.w_range*3, valinit=self.w_reset)
        w1 = Slider(w1_bar, 'w1', -self.w_range, self.w_range, valinit=self.w_reset)
        
        # 滑动时调用函数
        def update(val):
            # 获取更新数据
            l.set_ydata(self.activate11(w1.val, w0.val))
            fig.canvas.draw_idle()

        # 滑动时调用更新函数
        w0.on_changed(update)
        w1.on_changed(update)

        # reset 按钮
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        # 绑定 reset 函数
        def reset(event):
            w0.reset()
            w1.reset()
        button.on_clicked(reset)

        rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
        radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

        # 更新颜色
        def colorfunc(label):
            l.set_color(label)
            fig.canvas.draw_idle()
        radio.on_clicked(colorfunc)
        plt.show()

    def activate111(self, w11, w10, w21, w20):
        z = w21*(self.activate(w11 * self.x1 + w10)) + w20
        return self.activate(z)

    #   w11*x1 + w10  a1*w21 + w20 
    # x1----------->a1--------->y1
    def plot_neuron_111(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.35)
        plt.title('$%s[w_{21}%s(w_{11}x+w_{10})+w_{20}]$' % (self.acttype, self.acttype))
        
        # 使用默认值绘制图像
        s = self.activate111(self.w_reset, self.w_reset, self.w_reset, self.w_reset)
        l, = plt.plot(self.x1, s, lw=2, color='red')
        # x 轴范围和 y 轴范围
        plt.axis([-self.x_range, self.x_range, 0, 1])
        
        # 滑动条背景颜色
        axcolor = 'white'
        w10_bar = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
        w11_bar = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
        w20_bar = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        w21_bar = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

        # 绑定滑动条和 reset 数值
        w10 = Slider(w10_bar, 'w10', -self.w_range, self.w_range, valinit=self.w_reset)
        w11 = Slider(w11_bar, 'w11', -self.w_range, self.w_range, valinit=self.w_reset)
        w20 = Slider(w20_bar, 'w20', -self.w_range, self.w_range, valinit=self.w_reset)
        w21 = Slider(w21_bar, 'w21', -self.w_range, self.w_range, valinit=self.w_reset)
        
        # 滑动时调用函数
        def update(val):
            # 获取更新数据
            l.set_ydata(self.activate111(w11.val, w10.val, w21.val, w20.val))
            fig.canvas.draw_idle()

        # 滑动时调用更新函数
        w10.on_changed(update)
        w11.on_changed(update)
        w20.on_changed(update)
        w21.on_changed(update)
        
        # reset 按钮
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        # 绑定 reset 函数
        def reset(event):
            w10.reset()
            w11.reset()
            w20.reset()
            w21.reset()
        button.on_clicked(reset)
        
        rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
        radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

        # 更新颜色
        def colorfunc(label):
            l.set_color(label)
            fig.canvas.draw_idle()
        radio.on_clicked(colorfunc)
        plt.show()

    def activate21(self, x1, x2, w1, w2, b):
        z = (w1 * x1 + w2 * x2) + b
        return self.activate(z)

    #   w11*x1 + w12*x2 + w10 
    # x1----------->y1
    # x2----------|
    def plot_neuron_21(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.24)
        plt.title('%s$%s(w_{1}x_1+w_{2}x_2+w_{0})$' % ('\t'*2,self.acttype))
        
        # 使用默认值绘制图像
        plt.xlabel('x1')
        plt.ylabel('x2')

        self.x_range = 2
        self.x1 = np.arange(-self.x_range, self.x_range, 0.1)
        x1, x2 = np.meshgrid(self.x1, self.x1)
        predict = self.activate21(x1, x2, self.w_reset, self.w_reset, 
                                 self.w_reset)
        ax.plot_surface(x1, x2, predict, rstride=1, cstride=1, cmap='hot', edgecolor='none')
        # x 轴范围和 y 轴范围
        plt.axis([-self.x_range, self.x_range, -self.x_range, self.x_range])
        ax.set_zlim([0, 1])

        ax0 = plt.axes([0.05, 0.6, 0.3, 0.3])

        # 滑动条背景颜色
        axcolor = 'white'
        w0_bar = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        w1_bar = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        w2_bar = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        
        # 绑定滑动条和 reset 数值
        w0 = Slider(w0_bar, 'w0', -self.w_range, self.w_range, valinit=self.w_reset)
        w1 = Slider(w1_bar, 'w1', -self.w_range, self.w_range, valinit=self.w_reset)
        w2 = Slider(w2_bar, 'w2', -self.w_range, self.w_range, valinit=self.w_reset)
        
        # 滑动时调用函数
        cmap = 'hot'
        def update(val):
            # 获取更新数据
            ax.collections.clear()
            predict = self.activate21(x1, x2, w1.val, w2.val, w0.val)
            ax.plot_surface(x1, x2, predict, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
            try:
                ax0.clear()
                ax0.contour(x1, x2, predict, 20, cmap='hot')
            except:
                pass
            
            fig.canvas.draw_idle()

        # 滑动时调用更新函数
        w0.on_changed(update)
        w1.on_changed(update)
        w2.on_changed(update)
        
        # reset 按钮
        resetax = plt.axes([0.80, 0.20, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        # 绑定 reset 函数
        def reset(event):
            w0.reset()
            w1.reset()
            w2.reset()
        button.on_clicked(reset)
        plt.show()

    def activate211(self, x1, x2, w11, w12, w10, w21, w20):
        z = self.activate(w11 * x1 + w12 * x2 + w10)
        return self.activate(z * w21 + w20)
    
    #   activate(w11*x1 + w12*x2 + w10) * w21 + w20 
    # x1----------->a1 ---> y1
    # x2----------|
    def plot_neuron_211(self):
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.26)
        plt.title('%s$%s[w_{21}%s(w_{11}x_1 + w_{12}x_2 + w_{10})+w_{20}]$' 
                  % ('\t'*4, self.acttype, self.acttype))
        
        # 使用默认值绘制图像
        plt.xlabel('x1')
        plt.ylabel('x2')

        self.x_range = 2
        self.x1 = np.arange(-self.x_range, self.x_range, 0.1)
        x1, x2 = np.meshgrid(self.x1, self.x1)
        predict = self.activate211(x1, x2, 0,0,0,0,0)
        ax.plot_surface(x1, x2, predict, rstride=1, cstride=1, cmap='hot', edgecolor='none')
        # x 轴范围和 y 轴范围
        plt.axis([-self.x_range, self.x_range, -self.x_range, self.x_range])
        ax.set_zlim([0, 1])

        ax0 = plt.axes([0.05, 0.6, 0.3, 0.3])

        # 滑动条背景颜色
        axcolor = 'white'
        y_offsett = 0.2
        w10_bar = plt.axes([0.25, y_offsett, 0.65, 0.03], facecolor=axcolor)
        w11_bar = plt.axes([0.25, y_offsett - 0.05, 0.65, 0.03], facecolor=axcolor)
        w12_bar = plt.axes([0.25, y_offsett - 0.1, 0.65, 0.03], facecolor=axcolor)
        w20_bar = plt.axes([0.25, y_offsett - 0.15, 0.65, 0.03], facecolor=axcolor)
        w21_bar = plt.axes([0.25, y_offsett - 0.2, 0.65, 0.03], facecolor=axcolor)
        
        # 绑定滑动条和 reset 数值
        w10 = Slider(w10_bar, 'w10', -self.w_range, self.w_range, valinit=self.w_reset)
        w11 = Slider(w11_bar, 'w11', -self.w_range, self.w_range, valinit=self.w_reset)
        w12 = Slider(w12_bar, 'w12', -self.w_range, self.w_range, valinit=self.w_reset)
        w20 = Slider(w20_bar, 'w20', -self.w_range, self.w_range, valinit=self.w_reset)
        w21 = Slider(w21_bar, 'w21', -self.w_range, self.w_range, valinit=self.w_reset)
        
        # 滑动时调用函数
        cmap = 'hot'
        def update(val):
            # 获取更新数据
            ax.collections.clear()
            predict = self.activate211(x1, x2, w11.val, w12.val, w10.val, w21.val, w20.val)
            ax.plot_surface(x1, x2, predict, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
            try:
                ax0.clear()
                ax0.contour(x1, x2, predict, 20, cmap='hot')
            except:
                pass
            
            fig.canvas.draw_idle()

        # 滑动时调用更新函数
        w10.on_changed(update)
        w11.on_changed(update)
        w12.on_changed(update)
        w20.on_changed(update)
        w21.on_changed(update)
        
        # reset 按钮
        resetax = plt.axes([0.80, 0.24, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        # 绑定 reset 函数
        def reset(event):
            w10.reset()
            w11.reset()
            w12.reset()
            w20.reset()
            w21.reset()
        button.on_clicked(reset)
        plt.show()

if __name__ == "__main__":
    pltnn = pltNN(type='sigmoid')
    #pltnn.plot_neuron_11()
    pltnn.plot_neuron_111()
    #pltnn.plot_neuron_21()
    #pltnn.plot_neuron_211()

