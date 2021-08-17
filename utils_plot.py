import matplotlib as mpl

default_res=60
from matplotlib.lines import Line2D
#import matplotlib.pyplot as plt
#import numpy as np
#import cv2 as cv

#from utils_pytorch.utils import search_dict, ensure_path


def norm_and_map(data, cmap='jet', return_min_max=False):
    #print(weight_r.shape)
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max==data_min:
        data_norm = np.zeros(data.shape, dtype=data.dtype)
        data_norm[:,:,:] = 0.5        
    else:
        data_norm = (data - data_min) / (data_max - data_min) # normalize to [0, 1]
    cmap_func = plt.cm.get_cmap(cmap)
    data_mapped = cmap_func(data_norm) # [N_num, res_x, res_y, (r,g,b,a)]

    if return_min_max:
        return data_mapped, data_min, data_max
    else:
        return data_mapped

def norm(data, ):
    # to be implemented
    return

def get_res_xy(res, width, height):
    if width>=height:
        res_x = res
        res_y = int( res * height / width )
    else:
        res_y = res
        res_x = int( res * width / height )
    return res_x, res_y

def get_int_coords(x, y, xy_range, res_x, res_y):
    #return int( (x / box_width + 0.5) * res_x ), int( (y / box_height+0.5) * res_y )
    x0, y0, x1, y1 = xy_range
    return int( ((x-x0)/(x1-x0)) * res_x ), int( (y-y0)/(y1-y0) * res_y )

def get_int_coords_np(points, xy_range, res_x, res_y):
    #return int( (x / box_width + 0.5) * res_x ), int( (y / box_height+0.5) * res_y )
    #print('points shape:'+str(points.shape))
    x0, y0, x1, y1 = xy_range
    pixel_half_x = (x1 - x0) / ( 2 * res_x )
    pixel_half_y = (y1 - y0) / ( 2 * res_y )
    pixel_width_x = (x1 - x0) / res_x
    pixel_width_y = (y1 - y0) / res_y
    x, y = points[:, 0], points[:, 1]
    #print(x.shape)
    #print(y.shape)
    #return np.stack( [ ( ((x-x0-pixel_half_x)/(x1-x0)) * (res_x-1) ).astype(np.int), ( (y-y0-pixel_half_y)/(y1-y0) * (res_y-1) ).astype(np.int) ], axis=1)
    coords_int =  np.stack( [ ( (x-x0-pixel_half_x)/pixel_width_x ), ( (y-y0-pixel_half_y)/pixel_width_y )], axis=1)
    return np.around(coords_int).astype(np.int) # np.astype(np.int) do floor, not round.
def get_float_coords(i, j, xy_range, res_x, res_y):
    #x0, y0, x1, y1 = sum(xy_range, [])
    x0, y0, x1, y1 = xy_range
    return i/res_x*(x1-x0) + x0, j/res_y*(y1-y0) + y0

def get_float_coords_np(points, xy_range, res_x, res_y):
    x0, y0, x1, y1 = xy_range
    pixel_half_x = (x1 - x0) / ( 2 * res_x )
    pixel_half_y = (y1 - y0) / ( 2 * res_y )
    x_float = points[:,0] / res_x*(x1-x0) + x0 + pixel_half_x
    y_float = points[:,1] / res_y*(y1-y0) + y0 + pixel_half_y
    return np.stack([x_float, y_float], axis=1)

def plot_polyline_cv(img, points, closed=False, color=(0,0,0), width=2, type=4, xy_range=[[0.0,0.0],[1.0,1.0]]): #points:[point_num, (x,y)]
    if isinstance(points, list):
        points = np.array(points)

    point_num = points.shape[0]
    if closed:
        line_num = points.shape[0] - 1
    else:
        line_num = points.shape[0]

    x_res, y_res = img.shape[0], img.shape[1]

    for i in range(line_num):
        point_0 = get_int_coords(points[i%point_num][0], points[i%point_num][1], xy_range, x_res, y_res)
        point_1 = get_int_coords(points[(i+1)%point_num][0], points[(i+1)%point_num][1], xy_range, x_res, y_res)
        cv.line(img, point_0, point_1, color, width, type)

def plot_polyline_plt(ax, points, color=(0.0,0.0,0.0), width=2, closed=False): #points:[point_num, (x,y)]
    point_num = points.shape[0]
    if closed:
        plot_num = point_num
    else:
        plot_num = point_num - 1
    if isinstance(color, dict):
        method = search_dict(color, ['method', 'mode'])
        if method in ['start-end']:
            color_start = np.array(color['start'], dtype=np.float)
            color_end = np.array(color['end'], dtype=np.float)
            for i in range(plot_num):
                xs, ys = [ points[i][0], points[(i+1)%point_num][0]], [ points[i][1], points[(i+1)%point_num][1]]
                ratio = i / ( plot_num - 1 )
                color_now = tuple( ratio * color_end + (1.0 - ratio) * color_start )
                ax.add_line(Line2D( xs, ys, linewidth=width, color=color_now ))
        elif method in ['given']:
            color = search_dict(color, ['content', 'data'])
            for i in range(plot_num):
                xs, ys = [ points[i][0], points[(i+1)%point_num][0]], [ points[i][1], points[(i+1)%point_num][1]]
                ax.add_line(Line2D( xs, ys, linewidth=width, color=color[i] ))           
        else:
            raise Exception('plot_polyline_plt: invalid color mode:'+str(method))
        
    elif isinstance(color, tuple):
        for i in range(plot_num):
            xs, ys = [ points[i][0], points[(i+1)%point_num][0]], [ points[i][1], points[(i+1)%point_num][1]]
            ax.add_line(Line2D( xs, ys, linewidth=width, color=color ))
    else:
        raise Exception('plot_polyline_plt: invalid color mode:'+str(method))
    if isinstance(points, list):
        points = np.array(points)

plot_polyline = plot_polyline_plt

def plot_matrix(ax, data, save=True, save_path='./', save_name='matrix_plot.png', title=None, colorbar=True):
    im = ax.imshow(data)
    
    
    if title is not None:
        ax.set_title(title)
    ax.axis('off')

    if colorbar:
        max = np.max(data)
        min = np.min(data)
        

    if save:
        ensure_path(save_path)
        plt.savefig(save_path + save_name)

    return

def plot_line(img, points, line_color=(0,0,0), line_width=2, line_type=4, xy_range=[[0.0,0.0],[1.0,1.0]]):
    x_res, y_res = img.shape[0], img.shape[1]
    point_0 = get_int_coords(points[0][0], points[0][1], xy_range, x_res, y_res)
    point_1 = get_int_coords(points[1][0], points[1][1], xy_range, x_res, y_res)
    cv.line(img, point_0, point_1, line_color, line_width, line_type)

def get_colors(num=5):
    interval = 256 / num
    pos_now = 0.0
    colors = []
    for i in range(num):
        pos_now += interval
        colors.append(color_wheel(int(pos_now)))
    return colors

def color_wheel(pos): #生成横跨0-255个位置的彩虹颜色.  
    pos = pos % 255
    if pos < 85:
        return (pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return (0, pos * 3, 255 - pos * 3)

def plot_images(imgs, col_num):
    img_num = len(images)
    row_num = img_num // col_num
    if img_num%col_num>0:
        row_num += 1
    fig, axes = plt.subplots(row_num, col_num)

def cat_imgs_h(imgs, col_num=10, space_width=4):
    ''' Concat image horizontally with spacer '''
    space_col = np.ones([imgs.shape[1], space_width, imgs.shape[3]], dtype=np.uint8) * 255
    imgs_cols = []

    img_num = img.shape[0]
    if img_num < col_num:
        imgs = np.concatenate( [imgs, np.ones([imgs.shape[1], imgs.shape[2], imgs.shape[3]], dtype=np.uint8)*255] , axis=0)
    
    imgs_cols.append(space_col)
    for i in range(col_num):
        imgs_cols.append(imgs[i])
        imgs_cols.append(space_col)
    return np.concatenate(imgs_cols, axis=0)

def cat_imgs(imgs, col_num=10, space_width=4): # images: [num, width, height, channel_num], np.uint8
    img_num = imgs.shape[0]
    row_num = img_num // col_num
    if img_num%col_num>0:
        row_num += 1

    space_row = np.zeros([space_width, image.shape[0]*col_num + space_width*(col_num+1), imgs.shape[3]], dtype=np.uint8)
    space_row[:,:,:] = 255

    imgs_rows = []

    imgs_rows.append(space_row)
    for row_index in range(row_num):
        if (row_index+1)*col_num>img_num:
            imgs_row = imgs[ row_index*col_num : -1]
        else:
            imgs_row = imgs[ row_index*col_num : (row_index+1)*col_num ]
        imgs_row = cat_imgs_h(imgs_row, col_num, spacer_size)        
        imgs_rows.append(imgs_row)
        #if row_index != row_num-1:
        imgs_rows.append(space_row)

    return np.concatenate(imgs_rows, axis=1)



'''
def concat_images(images, image_width, spacer_size=4):
    # Concat image horizontally with spacer
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1: # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret

def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    # Concat images in rows
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret
'''