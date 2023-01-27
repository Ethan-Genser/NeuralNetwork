from graphics import *
win = GraphWin(width = 1920, height = 1080) # create a window

def ToHexColor(byte):
    return "#" + hex(byte)[2:] + hex(byte)[2:] + hex(byte)[2:]

def DrawNeuron(x, y, value):
    circle = Circle(Point(x, y), 20)
    color = ToHexColor(value)
    circle.setFill(color)
    circle.draw(win)

def DrawWieght(x1, y1, x2, y2, value):
    line = Line(Point(x1, y1), Point(x2, y2))
    color = ToHexColor(value)
    line.setOutline(color)
    line.setWidth(5)
    line.draw(win)
    pass

def DrawLayer(x, y, neurons, weights):
    pass

def DrawImg(x, y, img):
    pass

def Draw():
    pass

win.getMouse() # pause before closing