import numpy as np
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.core.fromnumeric import shape

default_res=60

import utils_torch
from utils_torch.utils import EnsurePath, ToNpArray, ToList
from utils_torch.attrs import *
from enum import Enum

ColorPlt = utils_torch.json.EmptyPyObj().FromDict({
    "White": (1.0, 1.0, 1.0),
    "Black": (0.0, 0.0, 0.0),
    "Red":   (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue":  (0.0, 0.0, 1.0),
})

def PlotLinesPlt(ax, XYsStart, XYsEnd=None, Width=1.0, Color=ColorPlt.Black):
    if XYsEnd is None:
        Edges = ToNpArray(XYsStart)
        XYsStart = Edges[:, 0]
        XYsEnd = Edges[:, 1]
    else:
        XYsStart = ToNpArray(XYsStart)
        XYsEnd = ToNpArray(XYsEnd)
    LineNum = XYsStart.shape[0]
    for Index in range(LineNum):
        X = [XYsStart[Index, 0], XYsEnd[Index, 0]]
        Y = [XYsStart[Index, 1], XYsEnd[Index, 1]]
        ax.add_line(Line2D(X, Y, linewidth=Width, color=Color))
PlotLines = PlotLinesPlt

def SetHeightWidthRatio(ax, ratio):
    ax.set_aspect(ratio)

def PlotLineAndMarkVerticesXY(ax, PointStart, PointEnd, Width=1.0, Color=ColorPlt.Black):
    PlotLinePlt(ax, PointStart, PointEnd, Width, Color)
    PlotPointAndMarkXY(ax, PointStart)
    PlotPointAndMarkXY(ax, PointEnd)

def PlotArrowAndMarkVerticesXY(ax, PointStart, PointEnd, Width=0.001, Color=ColorPlt.Black):
    PlotArrowFromVertexPairsPlt(ax, PointStart, PointEnd, Width=Width, Color=Color)
    PlotPointAndMarkXY(ax, PointStart)
    PlotPointAndMarkXY(ax, PointEnd)

def PlotLinePlt(ax, PointStart, PointEnd, Width=1.0, Color=ColorPlt.Black, Style="-"):
    # @param Width: Line Width in points(?pixels)
    X = [PointStart[0], PointEnd[0]]
    Y = [PointStart[1], PointEnd[1]]
    line = ax.add_line(Line2D(X, Y))
    line.set_linewidth(Width)
    line.set_color(Color)
    line.set_linestyle(Style)
PlotLine = PlotLinePlt

def PlotDashedLinePlt(ax, PointStart, PointEnd, Width=1.0, Color=ColorPlt.Black):
    PlotLinePlt(ax, PointStart, PointEnd, Width, Color, Style=(0, (5, 10)))
PlotDashedLine = PlotDashedLinePlt

def PlotPointAndAddText(ax, XY, PointColor=ColorPlt.Blue, Text="TextOnPoint", TextColor=None):
    if TextColor is None:
        TextColor = PointColor
    PlotPoint(ax, XY, PointColor)
    PlotText(ax, XY, Text, TextColor)

def PlotText(ax, XY, Text, Color=ColorPlt.Blue):
    ax.text(XY[0], XY[1], Text, color=Color)

def PlotPoint(ax, XY, Color=ColorPlt.Blue):
    ax.scatter([XY[0]], [XY[1]], color=Color)

def PlotPointsPltNp(ax, Points, Color=ColorPlt.Blue):
    Points = ToNpArray(Points)
    ax.scatter(Points[:, 0], Points[:, 1], color=Color)
    return
PlotPoints = PlotPointsPltNp

def PlotDirectionsOnEdges(ax, Edges, Directions, **kw):
    Edges = ToNpArray(Edges)
    Directions = ToNpArray(Directions)
    EdgesLength = utils_torch.geometry2D.Vectors2Lengths(utils_torch.geometry2D.VertexPairs2Vectors(Edges))
    ArrowsLength = 0.2 * np.median(EdgesLength, keepdims=True)
    Directions = utils_torch.geometry2D.Vectors2GivenLengths(Directions, ArrowsLength)
    MidPoints = utils_torch.geometry2D.Edges2MidPointsNp(Edges)
    PlotArrows(ax, MidPoints - 0.5 * Directions, Directions, **kw)

def PlotDirectionOnEdge(ax, Edge, Direction, **kw):
    PlotDirectionsOnEdges(ax, [Edge], [Direction], **kw)

def Map2Colors(data, ColorMap="jet", Method="MinMax", Alpha=False):
    data = ToNpArray(data)
    if Method in ["MinMax"]:
        dataMin, dataMax = np.min(data), np.max(data)
        if dataMin == dataMax:
            dataNormed = np.zeros(data.shape, dtype=data.dtype)
            dataNormed[:,:,:] = 0.5
        else:
            dataNormed = (data - dataMin) / (dataMax - dataMin) # normalize to [0, 1]
        dataMapped = ParseColorMapPlt(ColorMap)(dataNormed) # [*data.Shape, (r,g,b,a)]
    else:
        raise Exception()

    if not Alpha:
        dataMapped = eval("dataMapped[%s 0:3]"%("".join(":," for _ in range(len(data.shape)))))

    return dataMapped
Map2Color = Map2Colors

def MapColors2Colors(dataColored, ColorMap="jet", Range="MinMax"):
    return # to be implemented

def norm(data, ):
    # to be implemented
    return

def ParseResolutionXY(Resolution, Width, Height):
    if Width >= Height:
        ResolutionX = Resolution
        ResolutionY = int( Resolution * Height / Width )
    else:
        ResolutionY = Resolution
        ResolutionX = int( res * Width / Height )
    return ResolutionX, ResolutionY

def Floats2PixelIndex(x, y, BoundaryBox, ResolutionX, ResolutionY):
    #return int( (x / box_Width + 0.5) * ResolutionX ), int( (y / box_Height+0.5) * ResolutionY )
    x0, y0, x1, y1 = BoundaryBox
    return int( ((x-BoundaryBox.xMin)/(x1-x0)) * ResolutionX ), int( (y-y0)/(y1-y0) * ResolutionY )

def XY2PixelIndex(points, BoundaryBox, ResolutionX, ResolutionY):
    #return int( (x / box_Width + 0.5) * ResolutionX ), int( (y / box_Height+0.5) * ResolutionY )
    #print('points shape:'+str(points.shape))
    x0, y0, x1, y1 = BoundaryBox
    pixel_half_x = (x1 - x0) / ( 2 * ResolutionX )
    pixel_half_y = (y1 - y0) / ( 2 * ResolutionY )
    pixel_Width_x = (x1 - x0) / ResolutionX
    pixel_Width_y = (y1 - y0) / ResolutionY
    x, y = points[:, 0], points[:, 1]
    #print(x.shape)
    #print(y.shape)
    #return np.stack( [ ( ((x-x0-pixel_half_x)/(x1-x0)) * (ResolutionX-1) ).astype(np.int), ( (y-y0-pixel_half_y)/(y1-y0) * (ResolutionY-1) ).astype(np.int) ], axis=1)
    coords_int =  np.stack( [ ( (x-x0-pixel_half_x)/pixel_Width_x ), ( (y-y0-pixel_half_y)/pixel_Width_y )], axis=1)
    return np.around(coords_int).astype(np.int) # np.astype(np.int) do floor, not round.
def PixelIndex2XY(xIndex, yIndex, BoundaryBox, ResolutionX, ResolutionY):
    x0, y0, x1, y1 = BoundaryBox
    return xIndex * (BoundaryBox.xMax - BoundaryBox.xMin) / ResolutionX + BoundaryBox.xMin, yIndex / ResolutionY * (BoundaryBox.yMax - BoundaryBox.yMin) + BoundaryBox.yMin

def PixelIndices2XYs(Points, BoundaryBox, ResolutionX, ResolutionY):
    x0, y0, x1, y1 = BoundaryBox
    pixel_half_x = (x1 - x0) / ( 2 * ResolutionX )
    pixel_half_y = (y1 - y0) / ( 2 * ResolutionY )
    Xs = Points[:,0] / ResolutionX * (x1-x0) + x0 + pixel_half_x
    Ys = Points[:,1] / ResolutionY * (y1-y0) + y0 + pixel_half_y
    return np.stack([Xs, Ys], axis=1)

def PlotXYs(ax, XYs, XYsMark=None):
    if XYsMark is None:
        XYsMark = XYs
    for Index, XY in enumerate(XYs):
        PlotPointAndAddText(ax, XY, Text="(%.2f, %.2f)"%(XYsMark[Index][0], XYsMark[Index][1]))

def PlotPointAndMarkXY(ax, XY):
    PlotPointAndAddText(ax, XY, Text="(%.2f, %.2f)"%(XY[0], XY[1]))

def ParseColorPlt(Color):
    if isinstance(Color, tuple):
        if len(Color)==3:
            return Color
        else:
            raise Exception()
    elif isinstance(Color, str):
        if hasattr(ColorPlt, Color):
            return getattr(ColorPlt, Color)
        else:
            return Color
    else:
        raise Exception()

def ParseColorMapPlt(ColorMap):
    if isinstance(ColorMap, str):
        return plt.get_cmap(ColorMap)
    else:
        raise Exception()

def PlotArrows(ax, XYsStart, dXYs, Color=ColorPlt.Red):
    PlotNum = len(XYsStart)
    for Index in range(PlotNum):
        PlotArrowPlt(ax, XYsStart[Index], dXYs[Index], Color=Color)

def PlotArrowFromVertexPairsPlt(ax, XYStart, XYEnd, Width=0.001, Color=ColorPlt.Red, SizeScale=1.0):
    XYStart = ToNpArray(XYStart)
    XYEnd = ToNpArray(XYEnd)
    PlotArrowPlt(ax, XYStart, XYEnd - XYStart, Width=Width, Color=Color, SizeScale=SizeScale)

def PlotArrowPlt(ax, XYStart, dXY, Width=0.001, HeadWith=0.05, HeadLength=0.1, Color=ColorPlt.Red, SizeScale=None):
    Color = ParseColorPlt(Color)
    XYStart = ToList(XYStart)
    dXY = ToList(dXY)
    if SizeScale is not None:
        Width = 0.001 * SizeScale
        HeadWith = 0.05 * SizeScale
        HeadLength = 0.1 * SizeScale
    else:
        SizeScale = 1.0
    ax.arrow(*XYStart, *dXY, 
        width=Width * SizeScale,
        head_width=HeadWith * SizeScale,
        head_length=HeadLength * SizeScale,
        facecolor=Color,
        edgecolor=Color
    )

def PlotPolyLineFromVerticesPlt(ax, Points, Color=ColorPlt.Black, Width=2.0, Closed=False):
    # @param Points: np.ndarray with shape [PointNum, (x,y)]
    Points = ToList(Points)
    PointNum = len(Points)
    if Closed:
        LineNum = PointNum + 1
    else:
        LineNum = PointNum
        #Points = np.concatenate((Points, Points[-1, :][np.newaxis, :]), axis=0)

    if isinstance(Color, np.ndarray):
        pass
    else:
        for Index in range(LineNum):
            utils_torch.AddLog("%s %s"%(Points[Index], Points[(Index + 1)%PointNum]))
            PlotLinePlt(ax, Points[Index], Points[(Index + 1)%PointNum], Color=Color, Width=Width)

PlotPolyLine = PlotPolyLineFromVerticesPlt
    # if isinstance(color, dict):
    #     method = search_dict(color, ['method', 'mode'])
    #     if method in ['start-end']:
    #         color_start = np.array(color['start'], dtype=np.float)
    #         color_end = np.array(color['end'], dtype=np.float)
    #         for i in range(plot_num):
    #             xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #             ratio = i / ( plot_num - 1 )
    #             color_now = tuple( ratio * color_end + (1.0 - ratio) * color_start )
    #             ax.add_line(Line2D(xs, ys, linewidth=Width, color=color_now ))
    #     elif method in ['given']:
    #         color = search_dict(color, ['content', 'data'])
    #         for i in range(plot_num):
    #             xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #             ax.add_line(Line2D(xs, ys, linewidth=Width, color=color[i] ))           
    #     else:
    #         raise Exception('PlotPolyLinePlt: invalid color mode:'+str(method))
        
    # elif isinstance(color, tuple):
    #     for i in range(plot_num):
    #         xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #         ax.add_line(Line2D(xs, ys, linewidth=Width, color=color ))
    # else:
    #     raise Exception('PlotPolyLinePlt: invalid color mode:'+str(method))

def PlotPolyLineFromVerticesCV(img, points, closed=False, Color=(0,0,0), Width=2, Type=4, BoundaryBox=[[0.0,0.0],[1.0,1.0]]): #points:[PointNum, (x,y)]
    if isinstance(points, list):
        points = np.array(points)

    PointNum = points.shape[0]
    if closed:
        line_num = points.shape[0] - 1
    else:
        line_num = points.shape[0]

    x_res, y_res = img.shape[0], img.shape[1]

    for i in range(line_num):
        point_0 = Getint_coords(points[i%PointNum][0], points[i%PointNum][1], BoundaryBox, x_res, y_res)
        point_1 = Getint_coords(points[(i+1)%PointNum][0], points[(i+1)%PointNum][1], BoundaryBox, x_res, y_res)
        cv.line(img, point_0, point_1, color, Width, type)

def PlotMatrix(ax, data, Save=True, SavePath='./MatrixPlot.png', title=None, colorbar=True):
    im = ax.imshow(data)

    if title is not None:
        ax.set_title(title)
    ax.axis('off')

    if colorbar:
        max = np.max(data)
        min = np.min(data)
        
    if Save:
        utils_torch.EnsureFileDir(SavePath)
        plt.savefig(SavePath)
    return

def PlotLineCv(img, points, line_color=(0,0,0), line_Width=2, line_type=4, BoundaryBox=[[0.0,0.0],[1.0,1.0]]):
    x_res, y_res = img.shape[0], img.shape[1]
    point_0 = Getint_coords(points[0][0], points[0][1], BoundaryBox, x_res, y_res)
    point_1 = Getint_coords(points[1][0], points[1][1], BoundaryBox, x_res, y_res)
    cv.line(img, point_0, point_1, line_color, line_Width, line_type)

def GetRandomColors(Num=5):
    interval = 256 / Num
    pos_now = 0.0
    colors = []
    for i in range(Num):
        pos_now += interval
        colors.append(ColorWheel(int(pos_now)))
    return colors

GetColors = GetTypicalColors = GetRandomColors

def ColorWheel(Index): #生成横跨0-255个位置的彩虹颜色.  
    Index = Index % 255
    if Index < 85:
        return (Index * 3, 255 - Index * 3, 0)
    elif Index < 170:
        Index -= 85
        return (255 - Index * 3, 0, Index * 3)
    else:
        Index -= 170
        return (0, Index * 3, 255 - Index * 3)

def PlotImages(imgs, ColNum):
    img_num = len(images)
    RowNum = img_num // ColNum
    if img_num%ColNum>0:
        RowNum += 1
    fig, axes = plt.subplots(RowNum, ColNum)

def cat_imgs_h(imgs, ColNum=10, space_Width=4):
    ''' Concat image horizontally with spacer '''
    space_col = np.ones([imgs.shape[1], space_Width, imgs.shape[3]], dtype=np.uint8) * 255
    imgs_cols = []

    img_num = img.shape[0]
    if img_num < ColNum:
        imgs = np.concatenate( [imgs, np.ones([imgs.shape[1], imgs.shape[2], imgs.shape[3]], dtype=np.uint8)*255] , axis=0)
    
    imgs_cols.append(space_col)
    for i in range(ColNum):
        imgs_cols.append(imgs[i])
        imgs_cols.append(space_col)
    return np.concatenate(imgs_cols, axis=0)

def cat_imgs(imgs, ColNum=10, space_Width=4): # images: [num, Width, Height, channel_num], np.uint8
    img_num = imgs.shape[0]
    RowNum = img_num // ColNum
    if img_num%ColNum>0:
        RowNum += 1

    space_row = np.zeros([space_Width, image.shape[0]*ColNum + space_Width*(ColNum+1), imgs.shape[3]], dtype=np.uint8)
    space_row[:,:,:] = 255

    imgs_rows = []

    imgs_rows.append(space_row)
    for row_index in range(RowNum):
        if (row_index+1)*ColNum>img_num:
            imgs_row = imgs[ row_index*ColNum : -1]
        else:
            imgs_row = imgs[ row_index*ColNum : (row_index+1)*ColNum ]
        imgs_row = cat_imgs_h(imgs_row, ColNum, spacer_size)        
        imgs_rows.append(imgs_row)
        #if row_index != RowNum-1:
        imgs_rows.append(space_row)

    return np.concatenate(imgs_rows, axis=1)

def ParseRowColNum(PlotNum, RowNum, ColNum):
    # @param ColNum: int. Column Number.
    if RowNum is None and ColNum is not None:
        RowNum = PlotNum // ColNum
        if PlotNum % ColNum > 0:
            RowNum += 1
        return RowNum, ColNum
    elif RowNum is not None and ColNum is None:
        ColNum = PlotNum // RowNum
        if PlotNum % RowNum > 0:
            ColNum += 1
        return RowNum, ColNum
    elif RowNum is None and ColNum is None:
        ColNum = int(PlotNum ** 0.5)
        if ColNum == 0:
            ColNum = 1
        RowNum = PlotNum // ColNum
        if PlotNum % ColNum > 0:
            RowNum += 1
        return RowNum, ColNum
    else:
        if PlotNum != RowNum * ColNum:
            raise Exception('PlotNum: %d != RowNum %d x ColumnNum %d'%(PlotNum, RowNum, ColNum))
        else:
            return RowNum, ColNum

def CreateFigurePlt(PlotNum, RowNum=None, ColNum=None, Width=None, Height=None):
    RowNum, ColNum = ParseRowColNum(PlotNum, RowNum, ColNum)
    if Width is None and Height is None:
        Width = ColNum * 5.0 # inches
        Height = RowNum * 5.0 # inches
    elif Width is not None and Height is not None:
        pass
    else:
        raise Exception()
    fig, axes = plt.subplots(nrows=RowNum, ncols=ColNum, figsize=(Width, Height))
    return fig, axes
CreateFiture = CreateFigurePlt

def GetAxRowColNum(axes):
    if isinstance(axes, np.ndarray):
        Shape = axes.shape
        if len(Shape)==1:
            return Shape[0], 1
        elif len(Shape)==2:
            return Shape[0], Shape[1]
        else:
            raise Exception()
    # elif isinstance(axes, mpl.axes._subplots.AxesSubplot): # Shape is [1, 1]:
    #     return 1, 1
    else:
        return 1, 1
        # raise Exception()
def GetAx(axes, Index=None, RowIndex=None, ColIndex=None):
    RowNum, ColNum = GetAxRowColNum(axes)

    if Index is not None:
        RowIndex = Index // ColNum
        ColIndex = Index % ColNum
    
    if RowNum>1 or ColNum>1:
        if RowNum==1 or ColNum==1:
            if RowIndex==0 or RowIndex is None:
                if isinstance(ColIndex, int):
                    return axes[ColIndex]
                else:
                    raise Exception()
            elif ColIndex==0 or ColIndex is None:
                if isinstance(RowIndex, int):
                    return axes[RowIndex]
                else:
                    raise Exception()
            else:
                raise Exception()
        else:
            if RowIndex is None:
                raise Exception()
            if ColIndex is None:
                raise Exception()
            return axes[RowIndex, ColIndex]
    elif RowNum==1 and ColNum==1:
        if not (RowIndex is None or RowIndex==0):
            raise Exception()
        if not (RowIndex is None or RowIndex==0):
            raise Exception()
        return axes
    else:
        raise Exception()
'''
def concat_images(images, image_Width, spacer_size=4):
    # Concat image horizontally with spacer
    spacer = np.ones([image_Width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1: # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret

def concat_images_in_rows(images, row_size, image_Width, spacer_size=4):
    # Concat images in rows
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_Width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_Width, spacer_size)
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