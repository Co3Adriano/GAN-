import wradlib as wrl
import matplotlib.pyplot as pl
import warnings
import pandas 
import numpy as np
import matplotlib
#load radolan files
filename = ('C:/Users/Adria/Desktop/KI_Regen/RadoLanDataSet/2019/raa01-rw_10000-1901011550-dwd---bin')
print(filename)

rw_data, rw_attr = wrl.io.radolan.read_radolan_composite(filename, missing = -9999, loaddata = True)
print("RW Attributes:")
for key, value in rw_attr.items():
    print(key + ':', value)
print("-----------------------------------------------------------------")

###GET COORD
radolan_grid_xy = wrl.georef.get_radolan_grid(900,900)
radolan_egrid_xy = wrl.georef.get_radolan_grid(1500,1400)
radolan_wgrid_xy = wrl.georef.get_radolan_grid(1100, 900)
x = radolan_grid_xy[:,:,0]
y = radolan_grid_xy[:,:,1]

xe = radolan_egrid_xy[:,:,0]
ye = radolan_egrid_xy[:,:,1]

xw = radolan_wgrid_xy[:,:,0]
yw = radolan_wgrid_xy[:,:,1]
###RADOLAN PLOT
def plot_radolan(data, attrs, grid, clabel=None):
    fig = pl.figure(figsize=(10,8))
    ax = fig.add_subplot(111, aspect='equal')
    x = grid[:,:,0]
    y = grid[:,:,1]
    pm = ax.pcolormesh(x, y, data, cmap='viridis')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label(clabel)
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('{0} Product\n{1}'.format(attrs['producttype'],
                                       attrs['datetime'].isoformat()))
    pl.xlim((x[0,0],x[-1,-1]))
    pl.ylim((y[0,0],y[-1,-1]))
    pl.grid(color='r')
    pl.savefig('raa01-rw_10000-1901011550-dwd---bin') #####SAVES PLOT FIGURE



rw_data = np.ma.masked_equal(rw_data, -9999)
figure1 = plot_radolan(rw_data, rw_attr, radolan_grid_xy, clabel='mm * h-1')
pl.show(figure1) ##show plot image

print(rw_attr)    
