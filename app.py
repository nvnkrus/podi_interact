# import os
# import sys
# import time
# import numpy as np
# from scipy import io, integrate, linalg, signal
# from scipy.sparse.linalg import eigs
import os

from zmq import Message
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
from pyvista import examples

import scipy.io as sio
import pickle
import streamlit as st 
import vtk
from itkwidgets import view
from ipywidgets import embed
import streamlit.components.v1 as components

print(os.getcwd())
print(os.listdir())
msg = os.listdir()

# mat_fname = os.path.join('datasets','dome1_v1.mat')
# mat_contents = sio.loadmat(mat_fname)

# msg = str(mat_contents['dome'].dtype)
# print(msg)

# giving a title
st.title('Interactive PODI apps')
st.success(msg)

reader = vtk.vtkNIFTIImageReader()
fname = 'brain.nii'
reader.SetFileName(fname)
reader.Update()


view_width = 1600
view_height = 1200

snippet = embed.embed_snippet(views=view(reader.GetOutput()))
html = embed.html_template.format(title="", snippet=snippet)
components.html(html, width=view_width, height=view_height)


# pv.set_jupyter_backend('ipyvtklink')
# sphere = pv.Sphere()

# # short example
# image = sphere.plot(jupyter_backend='ipyvtklink', return_cpos=False)

# # long example
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(sphere)
# plotter.show()
# plotter.show(jupyter_backend='ipyvtklink')

# st.pyplot(plotter)
# # Define a simple Gaussian surface
# n = 20
# x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
# y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
# xx, yy = np.meshgrid(x, y)
# A, b = 100, 100
# zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

# # Get the points as a 2D NumPy array (N by 3)
# points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]

# cloud = pv.PolyData(points)
# cloud.plot(point_size=15)

# fname = os.path.join('database','para','paraval.bin')
# paraval = binread(fname=fname)
# nparaval = size(paraval,0)

# ## Read FE mesh of the soft gripper
# fname1 = fullfile('database','mesh','gcrd.bin');
# fname2 = fullfile('database','mesh','bfaces.bin');
# gcrd = binread(fname1);		# FE nodal coordinates
# bfaces = binread(fname2);	# FE mesh of boundary surface
# nnode = gcrd.shape[0];		# total number of FE nodes
# sdof = 3*nnode; 			# 

# # available validating parameters
# fname1 = fullfile('database','para','paraval.bin');
# fname2 = fullfile('database','para','dispval.bin');
# fname3 = fullfile('database','para','misesval.bin');
# paraval = binread(fname1);
# dispval = binread(fname2);
# misesval = binread(fname3);

# def validate_singlecase(valID,paraval,dispval,misesval):
# 	paras = paraval[valID]
# 	np.savetxt("parameter.csv", paras.reshape(1,1), delimiter=",",fmt='%.4f')
# 	flag = os.system(os.path.join(os.getcwd(),"run_main.sh"))

# 	print('==============================================\n')
# 	print('valID %d: d = %.4f\n' %(valID, paras))

# 	## Abaqus solution
# 	disp_true = dispval[:,valID]
# 	mises_true = misesval[:,valID]

# 	## PODI-RBF solution
# 	fname1 = fullfile('result','drom.bin');
# 	fname2 = fullfile('result','mrom.bin');
# 	fname3 = fullfile('result','runTime.bin');
# 	disp_new = binread(fname1);
# 	mises_new = binread(fname2);
# 	runTime = binread(fname3)

# 	## evaluate the errors of displacement and stress
# 	# displacement
# 	err_rtne = rtne(disp_new,disp_true)*100
# 	err_rmse = rmse(disp_new,disp_true)

# 	print('Runtime %.4f ms' %(runTime))
# 	print('Speed %.4f Hz' %(1000/runTime))
# 	print('Displacement:\n')
# 	print('\tError L2: %.4f %%\n' %err_rtne)
# 	print('\tError rmse: %.4g \n' %err_rmse)
# 	print('\tMax value : Abaqus %.4g, PODI-RBF %.4g \n' %(np.max(np.abs(disp_true)),np.max(np.abs(disp_new))))

# 	# mises stress
# 	err_rtne = rtne(mises_new,mises_true)*100
# 	err_rmse = rmse(mises_new,mises_true)

# 	print('von Mises stress:\n')
# 	print('\tError L2: %.4f %%\n' %err_rtne)
# 	print('\tError rmse: %.4g \n' %err_rmse)
# 	print('\tMax value : Abaqus %.4g, PODI-RBF %.4g \n' 
# 		%(np.max(np.abs(mises_true)),np.max(np.abs(mises_new))))

# for valID in range(nparaval): # [0]: #
# 	validate_singlecase(valID,paraval,dispval,misesval)