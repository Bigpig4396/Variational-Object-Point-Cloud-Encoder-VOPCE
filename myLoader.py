from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ModelNetDataLoader import ModelNetDataLoader
import numpy as np

class RenderObj(object):
    def __init__(self, data, color, pos, rot):
        self.data = data
        self.color = color
        self.pos = pos
        self.rot = rot

class MyLoader(object):
    def __init__(self, name, n_sample):
        self.N = n_sample
        if name == 'ModelNet':
            self.loader = ModelNetDataLoader('./data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)

    def InitGL(self, Width, Height):

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(Width) / float(Height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def plot(self, render_list):
        global window
        global plot_list
        plot_list = render_list
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(200, 200)

        window = glutCreateWindow('OpenGL Python Cube')

        glutDisplayFunc(self.DrawGLScene)
        glutIdleFunc(self.DrawGLScene)
        # glutKeyboardFunc(keyPressed)
        self.InitGL(640, 480)
        glutMainLoop()

    def DrawGLScene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        global plot_list
        for k in range(len(plot_list)):
            glLoadIdentity()
            glTranslatef(plot_list[k].pos[0], plot_list[k].pos[1], plot_list[k].pos[2])

            glRotatef(plot_list[k].rot[0], 1.0, 0.0, 0.0)
            glRotatef(plot_list[k].rot[1], 0.0, 1.0, 0.0)
            glRotatef(plot_list[k].rot[2], 0.0, 0.0, 1.0)
            for i in range(len(plot_list[k].data)):
                # Draw Cube (multiple quads)
                glBegin(GL_POINTS)

                glColor3f(plot_list[k].color[0], plot_list[k].color[1], plot_list[k].color[2])
                glVertex3f(plot_list[k].data[i][0], plot_list[k].data[i][1], plot_list[k].data[i][2])
                glEnd()
        glutSwapBuffers()

    def normalize(self, data):
        my_max = np.max(data, axis=0)
        my_min = np.min(data, axis=0)
        my_mean = (my_max+my_min)/2
        my_offset = (my_max-my_min)/2
        data = (data-my_mean)/my_offset
        return data

    def get_obj(self, index, color, pos, rot):
        data = self.loader[index][0][:, 0:3]
        data = self.normalize(data)
        temp_obj = RenderObj(data, color, pos, rot)
        return temp_obj

if __name__ == '__main__':
    n_sample = 512
    my_loader = MyLoader('ModelNet', n_sample=n_sample)
    render_list = []
    render_obj = my_loader.get_obj(index=0, color=[1.0, 0.0, 1.0], pos=[-1.5, 1.0, -6.0], rot=[0.0, 0.0, 0.0])
    render_list.append(render_obj)
    render_obj = my_loader.get_obj(index=1, color=[1.0, 1.0, 0.0], pos=[1.5, 1.0, -6.0], rot=[0.0, 0.0, 0.0])
    render_list.append(render_obj)
    my_loader.plot(render_list)