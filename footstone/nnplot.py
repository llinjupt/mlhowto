from matplotlib import pyplot
from math import cos, sin, atan

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, color='black'):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, color=color)
        pyplot.gca().add_patch(circle)

class Axon():
    # start and end coordinates like style (x,y)
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def draw(self, lw=1, color='black'):
        line = pyplot.Line2D((self.start[0], self.end[0]), 
                             (self.start[1], self.end[1]), lw=lw, color=color)
        pyplot.gca().add_line(line)

    def draw_arrow(self):
        pyplot.annotate("", xy=self.end, xytext=self.start, arrowprops=dict(arrowstyle="->"))

class Layer_BT():
    def __init__(self, network, number_of_neurons):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance * (neuron_in_layer_space - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance
       
        return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None
    
    def __line_between_two_neurons(self, prev_neuron, neuron):
        angle = atan((prev_neuron.x - neuron.x) / float(prev_neuron.y - neuron.y))
        
        x_adjustment = (neuron_radius + neuron_axon_space) * sin(angle)
        y_adjustment = (neuron_radius + neuron_axon_space)* cos(angle)
        axon = Axon((neuron.x - x_adjustment, neuron.y - y_adjustment),
                    (prev_neuron.x + x_adjustment, prev_neuron.y + y_adjustment))
        axon.draw()

    def draw(self, layer_name=''):
        x = self.neurons[0].x
        y = self.neurons[0].y
        pyplot.text(x - 2.5, y, layer_name, fontsize=14,
                     verticalalignment="center",
                     horizontalalignment="center")
        
        for neuron in self.neurons:
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(previous_layer_neuron, neuron)

# draw neuron network from left to right
class Layer_LR():
    def __init__(self, network, number_of_neurons):
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        y = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(self.x, -y)
            neurons.append(neuron)
            y += vertical_distance
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return vertical_distance * (neuron_in_layer_space - number_of_neurons) / 2

    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return self.previous_layer.x + horizontal_distance
        
        return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, prev_neuron, neuron):
        angle = atan((neuron.y - prev_neuron.y) / float(neuron.x - prev_neuron.x))
        x_adjustment = (neuron_radius + neuron_axon_space) * cos(angle)
        y_adjustment = (neuron_radius + neuron_axon_space) * sin(angle)
        
        axon = Axon((prev_neuron.x + x_adjustment, prev_neuron.y + y_adjustment),
                    (neuron.x - x_adjustment, neuron.y - y_adjustment))
        axon.draw(color='gray')

    def draw(self, layer_name=''):
        x = self.neurons[0].x
        y = self.neurons[0].y
        pyplot.text(x, y + 1.5, layer_name, fontsize=14,
                     verticalalignment="center",
                     horizontalalignment="center")

        for neuron in self.neurons:
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(previous_layer_neuron, neuron)

# draw the network from left to right: 'h', bottom to top : 'v'
class NeuralNetwork():
    def __init__(self, direction='h'):
        self.layers = []
        self.layerclass = Layer_LR if direction == 'h' else Layer_BT
        
    def add_layer(self, number_of_neurons):
        layer = self.layerclass(self, number_of_neurons)
        self.layers.append(layer)

    def draw(self):
        layers = len(self.layers)
        
        if layers == 0:
            return
        
        if layers >= 1:
            self.layers[0].draw('Input')

        for layer in self.layers[1:-1]:
            layer.draw('Hidden')
            
        if layers >= 2:
            self.layers[-1].draw('Output')
        
        pyplot.axis('scaled')
        pyplot.xticks([])
        pyplot.yticks([])
        
        ax = pyplot.gca()
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('none')

        pyplot.show()

if __name__ == "__main__":
    vertical_distance = 3
    horizontal_distance = 6
    neuron_radius = 0.6
    neuron_axon_space = 0.3 * neuron_radius
    neuron_in_layer_space = 4
    
    network = NeuralNetwork()
    network.add_layer(2)
    network.add_layer(3)
    network.add_layer(2)
    network.add_layer(4)
    network.draw()
