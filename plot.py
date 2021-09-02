import numpy as np
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

default_res=60

from utils_torch.utils import EnsurePath
from enum import Enum

class ColorPlt(Enum):
    White = (1.0, 1.0, 1.0)
    Black = (0,0, 0.0, 0.0)
    Red = (1.0, 0.0, 0.0)
    Green = (0.0, 1.0, 0.0)
    Blue = (0.0, 0.0, 1.0)

def PlotLinesPltNp(ax, PointsStart, PointsEnd, Width=2.0, Color=ColorPlt.Black):
    LineNum = PointsStart.shape[0]
    for Index in range(LineNum):
        ax.add_line(Line2D(PointsStart[Index, :], PointsEnd[Index, :], linewidth=Width, Color=Color))
PlotLines = PlotLinesPltNp

def PlotLinePlt(ax, PointStart, PointEnd, Width=2.0, Color=ColorPlt.Black):
    # @param Width: Line width in points(?pixels)
    line = ax.add_line(Line2D(PointStart, PointEnd))
    line.set_linewidth(Width)
    line.set_color(Color)
PlotLine = PlotLinePlt

def PlotPointsNp(ax, Points, color=ColorPlt.Blue):
    ax.scatter(Points, color=color)
    return

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
    data_mapped = cmap_func(data_norm) # [N_num, ResolutionX, ResolutionY, (r,g,b,a)]

    if return_min_max:
        return data_mapped, data_min, data_max
    else:
        return data_mapped

def norm(data, ):
    # to be implemented
    return

def get_ResolutionXy(res, width, height):
    if width>=height:
        ResolutionX = res
        ResolutionY = int( res * height / width )
    else:
        ResolutionY = res
        ResolutionX = int( res * width / height )
    return ResolutionX, ResolutionY

def Floats2PixelIndex(x, y, BoundaryBox, ResolutionX, ResolutionY):
    #return int( (x / box_width + 0.5) * ResolutionX ), int( (y / box_height+0.5) * ResolutionY )
    x0, y0, x1, y1 = BoundaryBox
    return int( ((x-BoundaryBox.xMin)/(x1-x0)) * ResolutionX ), int( (y-y0)/(y1-y0) * ResolutionY )

def Xy2PixelIndex(points, BoundaryBox, ResolutionX, ResolutionY):
    #return int( (x / box_width + 0.5) * ResolutionX ), int( (y / box_height+0.5) * ResolutionY )
    #print('points shape:'+str(points.shape))
    x0, y0, x1, y1 = BoundaryBox
    pixel_half_x = (x1 - x0) / ( 2 * ResolutionX )
    pixel_half_y = (y1 - y0) / ( 2 * ResolutionY )
    pixel_width_x = (x1 - x0) / ResolutionX
    pixel_width_y = (y1 - y0) / ResolutionY
    x, y = points[:, 0], points[:, 1]
    #print(x.shape)
    #print(y.shape)
    #return np.stack( [ ( ((x-x0-pixel_half_x)/(x1-x0)) * (ResolutionX-1) ).astype(np.int), ( (y-y0-pixel_half_y)/(y1-y0) * (ResolutionY-1) ).astype(np.int) ], axis=1)
    coords_int =  np.stack( [ ( (x-x0-pixel_half_x)/pixel_width_x ), ( (y-y0-pixel_half_y)/pixel_width_y )], axis=1)
    return np.around(coords_int).astype(np.int) # np.astype(np.int) do floor, not round.
def get_float_coords(i, j, BoundaryBox, ResolutionX, ResolutionY):
    #x0, y0, x1, y1 = sum(BoundaryBox, [])
    x0, y0, x1, y1 = BoundaryBox
    return i/ResolutionX*(x1-x0) + x0, j/ResolutionY*(y1-y0) + y0

def get_float_coords_np(points, BoundaryBox, ResolutionX, ResolutionY):
    x0, y0, x1, y1 = BoundaryBox
    pixel_half_x = (x1 - x0) / ( 2 * ResolutionX )
    pixel_half_y = (y1 - y0) / ( 2 * ResolutionY )
    x_float = points[:,0] / ResolutionX*(x1-x0) + x0 + pixel_half_x
    y_float = points[:,1] / ResolutionY*(y1-y0) + y0 + pixel_half_y
    return np.stack([x_float, y_float], axis=1)


def plot_polyline_cv(img, points, closed=False, color=(0,0,0), width=2, type=4, BoundaryBox=[[0.0,0.0],[1.0,1.0]]): #points:[PointNum, (x,y)]
    if isinstance(points, list):
        points = np.array(points)

    PointNum = points.shape[0]
    if closed:
        line_num = points.shape[0] - 1
    else:
        line_num = points.shape[0]

    x_res, y_res = img.shape[0], img.shape[1]

    for i in range(line_num):
        point_0 = get_int_coords(points[i%PointNum][0], points[i%PointNum][1], BoundaryBox, x_res, y_res)
        point_1 = get_int_coords(points[(i+1)%PointNum][0], points[(i+1)%PointNum][1], BoundaryBox, x_res, y_res)
        cv.line(img, point_0, point_1, color, width, type)

def PlotPolyLineFromVerticesPlt(ax, Points, Color=ColorPlt.Black, width=2.0, Closed=False):
    # @param Points: np.ndarray with shape [PointNum, (x,y)]
    PointNum = Points.shape[0]
    if Closed:
        LineNum = PointNum + 1
    else:
        LineNum = PointNum
        #Points = np.concatenate((Points, Points[-1, :][np.newaxis, :]), axis=0)

    if isinstance(Color, np.ndarray):
        pass
    else:
        for Index in range(LineNum):
            PlotLinePlt(ax, Points[Index], Points[(Index + 1)%PointNum])

    # if isinstance(color, dict):
    #     method = search_dict(color, ['method', 'mode'])
    #     if method in ['start-end']:
    #         color_start = np.array(color['start'], dtype=np.float)
    #         color_end = np.array(color['end'], dtype=np.float)
    #         for i in range(plot_num):
    #             xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #             ratio = i / ( plot_num - 1 )
    #             color_now = tuple( ratio * color_end + (1.0 - ratio) * color_start )
    #             ax.add_line(Line2D( xs, ys, linewidth=width, color=color_now ))
    #     elif method in ['given']:
    #         color = search_dict(color, ['content', 'data'])
    #         for i in range(plot_num):
    #             xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #             ax.add_line(Line2D( xs, ys, linewidth=width, color=color[i] ))           
    #     else:
    #         raise Exception('PlotPolyLinePlt: invalid color mode:'+str(method))
        
    # elif isinstance(color, tuple):
    #     for i in range(plot_num):
    #         xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #         ax.add_line(Line2D( xs, ys, linewidth=width, color=color ))
    # else:
    #     raise Exception('PlotPolyLinePlt: invalid color mode:'+str(method))

def PlotMatrix(ax, data, save=True, save_path='./', save_name='matrix_plot.png', title=None, colorbar=True):
    im = ax.imshow(data)

    if title is not None:
        ax.set_title(title)
    ax.axis('off')

    if colorbar:
        max = np.max(data)
        min = np.min(data)
        

    if save:
        EnsurePath(save_path)
        plt.savefig(save_path + save_name)

    return

def plot_line(img, points, line_color=(0,0,0), line_width=2, line_type=4, BoundaryBox=[[0.0,0.0],[1.0,1.0]]):
    x_res, y_res = img.shape[0], img.shape[1]
    point_0 = get_int_coords(points[0][0], points[0][1], BoundaryBox, x_res, y_res)
    point_1 = get_int_coords(points[1][0], points[1][1], BoundaryBox, x_res, y_res)
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


def ImagesFile2GIFFile():
    return

ImagesFile2GIF = ImagesFile2GIFFile

def ImagesNp2GIFFile():
    return
ImagesNp2GIF = ImagesNp2GIFFile

'''
from PIL import Image 
from images2gif import writeGif
def create_gif_2():
    outfilename = "my.gif" # 转化的GIF图片名称          
    filenames = []         # 存储所需要读取的图片名称
    for i in range(100):   # 读取100张图片
        filename = path    # path是图片所在文件，最后filename的名字必须是存在的图片 
        filenames.append(filename)              # 将使用的读取图片汇总
    frames = []
    for image_name in filenames:                # 索引各自目录
        im = Image.open(image_name)             # 将图片打开，本文图片读取的结果是RGBA格式，如果直接读取的RGB则不需要下面那一步
        im = im.convert("RGB")                  # 通过convert将RGBA格式转化为RGB格式，以便后续处理 
        im = np.array(im)                       # im还不是数组格式，通过此方法将im转化为数组
        frames.append(im)                       # 批量化
    writeGif(outfilename, frames, duration=0.1, subRectangles=False) # 生成GIF，其中durantion是延迟，这里是1ms
'''