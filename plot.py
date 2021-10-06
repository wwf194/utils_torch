from os import EX_CANTCREAT
from re import L, X
import numpy as np
import scipy
import cv2 as cv
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_style("white")

default_res=60

import utils_torch
from utils_torch.utils import ToNpArray, ToList
from utils_torch.attrs import *
from enum import Enum

ColorPlt = utils_torch.EmptyPyObj().FromDict({
    "White": (1.0, 1.0, 1.0),
    "Black": (0.0, 0.0, 0.0),
    "Red":   (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue":  (0.0, 0.0, 1.0),
    "Gray":  (0.5, 0.5, 0.5),
    "Grey":  (0.5, 0.5, 0.5),
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

def ParsePointTypePlt(Type):
    if isinstance(Type, str):
        if Type in ["Circ", "Circle"]:
            return "o"
        elif Type in ["Triangle"]:
            return "^"
        else:
            raise Exception(Type)
    else:
        raise Exception(Type)

def PlotPoint(ax, XY, Color=ColorPlt.Blue, Type="Circle"):
    Color = ParseColorPlt(Color)
    Type = ParsePointTypePlt(Type)
    ax.scatter([XY[0]], [XY[1]], color=Color, marker=Type)

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

def Map2Color(
        data, ColorMap="jet", Method="MinMax", Alpha=False,
        dataForMap=None,
    ):
    data = ToNpArray(data)

    if Method in ["MinMax"]:
        if dataForMap is not None:
            if dataForMap.size > 0:
                dataMin, dataMax = np.nanmin(dataForMap), np.nanmax(dataForMap)
            else:
                dataMin, dataMax = np.NaN, np.NaN
        else:
            dataMin, dataMax = np.nanmin(data), np.nanmax(data)
        if dataMin == dataMax or not np.isfinite(dataMin) or not np.isfinite(dataMax):
            dataColored = np.full([*data.shape, 4], 0.5)
        else:
            dataNormed = (data - dataMin) / (dataMax - dataMin) # normalize to [0, 1]
            dataColored = ParseColorMapPlt(ColorMap)(dataNormed) # [*data.Shape, (r,g,b,a)]
    else:
        raise Exception()

    if not Alpha:
        dataColored = eval("dataColored[%s 0:3]"%("".join(":," for _ in range(len(data.shape)))))
        #dataColored = dataColored.take([0, 1, 2], axis=-1)
    return utils_torch.PyObj({
        "dataColored": dataColored,
        "Min": dataMin,
        "Max": dataMax
    })
Map2Colors = Map2Color

def MapColors2Colors(dataColored, ColorMap="jet", Range="MinMax"):
    return # to be implemented

def norm(data):
    # to be implemented
    return

def ParseResolutionXY(Resolution, Width, Height):
    if Width >= Height:
        ResolutionX = Resolution
        ResolutionY = int( Resolution * Height / Width )
    else:
        ResolutionY = Resolution
        ResolutionX = int(Resolution * Width / Height )
    return ResolutionX, ResolutionY

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

ParseColorMap = ParseColorMapPlt

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
            # utils_torch.AddLog("(%.3f %.3f) (%.3f %.3f)"%(Points[Index][0], Points[Index][1], Points[(Index + 1)%PointNum][0], Points[(Index + 1)%PointNum][1]))
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

def ParseResolution(Width, Height, Resolution):
    if Width < Height:
        ResolutionX = Resolution
        ResolutionY = round(ResolutionX * Height / Width)
    else:
        ResolutionY = Resolution
        ResolutionX = round(ResolutionY * Width / Height)
    return ResolutionX, ResolutionY

def PlotPolyLineFromVerticesCV(img, points, closed=False, Color=(0,0,0), Width=2, Type=4, BoundaryBox=[[0.0,0.0],[1.0,1.0]]): #points:[PointNum, (x,y)]
    if isinstance(points, list):
        points = np.array(points)

    PointNum = points.shape[0]
    if closed:
        line_num = points.shape[0] - 1
    else:
        line_num = points.shape[0]

    ResolutionX, ResolutionY = img.shape[0], img.shape[1]

    for i in range(line_num):
        point_0 = utils_torch.geometry2D.XY2PixelIndex(points[i%PointNum][0], points[i%PointNum][1], BoundaryBox, ResolutionX, ResolutionY)
        point_1 = utils_torch.geometry2D.XY2PixelIndex(points[(i+1)%PointNum][0], points[(i+1)%PointNum][1], BoundaryBox, ResolutionX, ResolutionY)
        cv.line(img, point_0, point_1, Color, Width, type)

def SetAxRangeFromBoundaryBox(ax, BoundaryBox):
    ax.set_xlim(BoundaryBox.XMin, BoundaryBox.XMax)
    ax.set_ylim(BoundaryBox.YMin, BoundaryBox.YMax)
    #ax.set_xticks(np.linspace(BoundaryBox.XMin, BoundaryBox.XMax, 5))
    #ax.set_yticks(np.linspace(BoundaryBox.YMin, BoundaryBox.YMax, 5))

def SetAxRangeAndTicksFromBoundaryBox(ax, BoundaryBox):
    SetAxRangeFromBoundaryBox(ax, BoundaryBox)
    SetXTicksFloat(ax, BoundaryBox.XMin, BoundaryBox.XMax)
    SetYTicksFloat(ax, BoundaryBox.YMin, BoundaryBox.YMax)

def ParseIsMatrixDataColored(data):
    DimensionNum = utils_torch.GetDimensionNum(data)
    if DimensionNum==2: # [XNum, YNum]
        IsDataColored = False
    elif DimensionNum==3: # [XNum, YNum, (r, g, b) or (r, g, b, a)], already mapped to colors.
        IsDataColored = True
    else:
        raise Exception(DimensionNum)  
    return IsDataColored

def PlotMatrixWithColorBar(
        ax, data, IsDataColored=None, ColorMap="jet", ColorMethod="MinMax", 
        XYRange=None, Coordinate="Math", dataForColorMap=None, dataMask=None,
        Ticks=None, XLabel=None, YLabel=None, Title=None,
        ColorBarOrientation="Auto", ColorBarLocation=None,
        PixelHeightWidthRatio="Equal",
        ColorBarTitle=None, Save=False, SavePath=None, **kw
    ):
    if IsDataColored is None:
        IsDataColored = ParseIsMatrixDataColored(data)

    if not IsDataColored:
        if dataForColorMap is not None:
            dataMapResult = Map2Color(data, ColorMap, dataForMap=dataForColorMap) 
        else:
            dataMapResult = Map2Color(data, ColorMap)
        data = dataMapResult.dataColored
        Min = dataMapResult.Min
        Max = dataMapResult.Max
        kw.update({
            "Min": Min,
            "Max": Max,
        })
    else:
        pass

    if PixelHeightWidthRatio in ["FillAx"]:
        PixelHeightWidthRatio = "auto"
        axMatrix = ax
    elif PixelHeightWidthRatio in ["Auto", "auto"]:
        DataHeightWidthRatio = data.shape[0] / data.shape[1]
        if DataHeightWidthRatio > 5.0:
            # data.shape[0] * PixelHeight / (data.shape[1] * PixelWidth) = 5.0
            # data.shape[0] / data.shape[1] * PixelHeightWidthRatio = 5.0
            PixelHeightWidthRatio = 5.0 * data.shape[1] / data.shape[0]
            axMatrix = GetSubAx(ax, 0.4, 0.0, 0.2, 1.0)
        elif DataHeightWidthRatio < 0.2:
            PixelHeightWidthRatio = 0.2 * data.shape[1] / data.shape[0]
            axMatrix = GetSubAx(ax, 0.0, 0.4, 1.0, 0.2)
        else:
            PixelHeightWidthRatio = "equal"
            if data.shape[0] > data.shape[1]:
                WidthHeightRatio = 1.0 * data.shape[1] / data.shape[0]
                axMatrix = GetSubAx(ax, 0.5 - WidthHeightRatio / 2.0, 0.0, WidthHeightRatio, 1.0)
            else:
                HeightWidthRatio = 1.0 * data.shape[0] / data.shape[1]
                axMatrix = GetSubAx(ax, 0.0, 0.5 - HeightWidthRatio / 2.0, 1.0, HeightWidthRatio)
    else:
        axMatrix = ax

    ax.axis("off")
    PlotMatrix(
        axMatrix, data, True, ColorMap, XYRange, Coordinate=Coordinate, 
        PixelHeightWidthRatio=PixelHeightWidthRatio, Ticks=Ticks,
        dataMask=dataMask, Save=False, 
        XLabel=XLabel, YLabel=YLabel, Title=Title, **kw
    )
    
    # if ColorBarOrientation in ["Auto", "auto"]:
    #     if data.shape[0] / data.shape[1] > 1.5:
    #         ColorBarOrientation = "vertical"
    #     else:
    #         ColorBarOrientation = "horizontal"
    
    axColorBar = GetSubAx(ax, 1.10, 0.0, 0.1, 1.0)
    ColorBarOrientation = "vertical"
    PlotColorBarInAx(
        axColorBar, ColorMap=ColorMap, Method=ColorMethod, Orientation=ColorBarOrientation, 
        Location=ColorBarLocation, Title=ColorBarTitle, **kw
    )

    #SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    plt.tight_layout()
    SaveFigForPlt(Save, SavePath)

def PlotMatrix(
        ax, data, IsDataColored=None, 
        ColorMap="jet", XYRange=None, Coordinate="Math", dataMask=None, PixelHeightWidthRatio="Auto",
        XLabel=None, YLabel=None, Title=None, dataForColorMap=None, Ticks=None,
        Save=False, SavePath=None, Format="svg", **kw
    ):
    if IsDataColored is None:
        DimensionNum = utils_torch.GetDimensionNum(data)
        if DimensionNum==2: # [XNum, YNum]
            IsDataColored = False
        elif DimensionNum==3: # [XNum, YNum, (r, g, b) or (r, g, b, a)], already mapped to colors.
            IsDataColored = True
        else:
            raise Exception(DimensionNum)

    if not IsDataColored:
        dataMapResult = Map2Color(data, ColorMap, dataForMap=dataForColorMap)
        data = dataMapResult.dataColored

    if XYRange is not None:
        extent = [XYRange.XMin, XYRange.XMax, XYRange.YMin, XYRange.YMax]
    else:
        extent = [0.0, data.shape[1], data.shape[0], 0.0] # [Left, Right, Bottom, Top]

    if dataMask is not None:
        InsideMask = dataMask.astype(np.float32)
        OutsideMask = (~dataMask).astype(np.float32)
        maskColor = kw.setdefault("maskColor", "Gray")
        maskColor = ParseColorPlt(maskColor)
        data = InsideMask[:, :, np.newaxis] * data + OutsideMask[:, :, np.newaxis] * maskColor

    if Coordinate in ["Math"]:
        data = data.transpose(1, 0, 2)
        data = data[::-1, :, :]
    elif Coordinate in ["Fig", "Picture"]:
        pass

    if PixelHeightWidthRatio in ["FillAx"]:
        PixelHeightWidthRatio = "auto"
    elif PixelHeightWidthRatio in ["Auto", "auto"]:
        DataHeightWidthRatio = data.shape[0] / data.shape[1]
        if DataHeightWidthRatio > 5.0:
            # data.shape[0] * PixelHeight / (data.shape[1] * PixelWidth) = 5.0
            # data.shape[0] / data.shape[1] * PixelHeightWidthRatio = 5.0
            PixelHeightWidthRatio = 5.0 * data.shape[1] / data.shape[0]
        elif DataHeightWidthRatio < 0.2:
            PixelHeightWidthRatio = 0.2 * data.shape[1] / data.shape[0]
        else:
            PixelHeightWidthRatio = "equal"
    elif PixelHeightWidthRatio in ["Equal"]:
        PixelHeightWidthRatio = "equal"

    if Ticks is not None:
        if Ticks in ["int", "Int"]:
            SetXTicksInt(ax, extent[0], extent[1])
            SetYTicksInt(ax, extent[2], extent[3])
        elif Ticks in ["float", "Float",]:
            SetXTicksFloat(ax, extent[0], extent[1])
            SetYTicksFloat(ax, extent[2], extent[3])
        else:
            raise Exception(Ticks)
    Interpolation = kw.get("Interpolation")
    if Format in ["svg"]:
        Interpolation = 'none'

    ax.imshow(data, extent=extent, aspect=PixelHeightWidthRatio, interpolation=Interpolation)    
    SetMatplotlibParamToDefault() # ticks may disappear after imshow.

    SetAxisLocationForAx(ax, kw.get("XAxisLocation"), kw.get("YAxisLocation"))
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SaveFigForPlt(Save, SavePath)
    return ax

def PlotActivityAndDistributionAlongTime(
        axes, activity, activityPlot, 
        Title=None,
        Save=True, SavePath=None,
    ):
    BatchSize = activity.shape[0]
    TimeNum = activity.shape[1]
    ActivityNum = activity.shape[2]

    activity = utils_torch.ToNpArray(activity)
    activityPlot = utils_torch.ToNpArray(activityPlot).transpose(1, 0)
    if axes is None:
        #fig, axes = plt.subplots(nrows=1, ncols=2)
        fig, axes = CreateFigurePlt(2)
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1 = GetAx(axes, 0)
        ax2 = GetAx(axes, 1)

    dataForColorMap, dataMask = MaskOutInfOrNaN(activityPlot)
    PlotMatrixWithColorBar(
        ax1, activityPlot, dataForColorMap=dataForColorMap, 
        dataMask=dataMask, maskColor="Gray",
        XAxisLocation="top", PixelHeightWidthRatio="Auto",
        XLabel="Time Index", YLabel="Activity", Title="Visualization",
        Coordinate="Fig", Ticks="Int"
    )
    PlotMeanAndStdAlongTime(
        ax2, Xs=range(TimeNum), 
        #Mean=utils_torch.math.ReplaceNaNOrInfWithZero(np.nanmean(activity, axis=(0, 2))),
        #Std=utils_torch.math.ReplaceNaNOrInfWithZero(np.nanstd(activity, axis=(0, 2))),
        Mean = np.nanmean(activity, axis=(0, 2)),
        Std = np.nanstd(activity, axis=(0,2)),
        XLabel="Time Index", YLabel="Mean And Std", Title="Mean and Std Along Time",
        Save=False
    )
    if Title is not None:
        plt.suptitle(Title)
    
    plt.tight_layout()
    SaveFigForPlt(Save, SavePath)
    return

def PlotWeightChange(axes=None, weights=None):
    if axes is None:
        fig, axes = CreateFigurePlt(3)

    return

def PlotWeightAndDistribution(axes=None, weight=None, Name=None, SavePath=None):
    if axes is None:
        fig, axes = CreateFigurePlt(2)
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1 = GetAx(axes, 0)
        ax2 = GetAx(axes, 1)

    plt.suptitle(Name)
    weight = utils_torch.ToNpArray(weight)
    _weight = weight
    weightForColorMap, _maskInfOrNaN = MaskOutInfOrNaN(_weight)

    # turn 1D weight to 2D shape
    DimentionNum = utils_torch.GetDimensionNum(weight)
    if DimentionNum == 1:
        weight, maskReshape = utils_torch.Line2Square(weight)
        XLabel, YLabel = "Dimension 0", "Dimension 0"
    else:
        maskReshape = None
        XLabel, YLabel = "Dimension 1", "Dimension 0"

    _weightForColorMap, maskInfOrNaN = MaskOutInfOrNaN(weight)
    
    dataMask = Merge2Mask(maskReshape, maskInfOrNaN)

    utils_torch.plot.PlotMatrixWithColorBar(
        ax1, weight, dataForColorMap=weightForColorMap, dataMask=dataMask,
        XAxisLocation="top", PixelHeightWidthRatio="Auto",
        Coordinate="Fig", Ticks="Int",
        Title="Visualization", XLabel=XLabel, YLabel=YLabel
    )
    
    #utils_torch.plot.PlotGaussianDensityCurve(axRight, weight) # takes too much time
    utils_torch.plot.PlotHistogram(
        ax2, weightForColorMap, Color="Black",
        XLabel="Connection Strength", YLabel="Ratio", Title="Distribution"
    )

    plt.suptitle("%s Shape:%s"%(Name, weight.shape))
    plt.tight_layout()
    if SavePath is None:
        SavePath = utils_torch.GetSaveDir + "weights/" + "%s.svg"%Name
    utils_torch.plot.SaveFigForPlt(SavePath=SavePath)
    return

def Merge2Mask(mask1, mask2):
    if mask1 is not None and mask2 is not None:
        return mask1 * mask2
    elif mask1 is None and mask2 is not None:
        return mask2
    elif mask1 is not None and mask2 is None:
        return mask1
    else:
        return None

def GetInfOrNaNMask(data):
    mask = np.isfinite(data)
    if round(np.sum(mask)) == mask.size:
        mask = None
    else:
        return mask

def MaskOutInfOrNaN(data, ReturnNoneIfNoInfOrNaN=True):
    mask = GetInfOrNaNMask(data)
    if mask is None:
        return data, None
    else:
        return data[mask], mask

def PlotLineCv(img, points, line_color=(0,0,0), line_Width=2, line_type=4, BoundaryBox=[[0.0,0.0],[1.0,1.0]]):
    ResolutionX, ResolutionY = img.shape[0], img.shape[1]
    point_0 = GetIntCoords(points[0][0], points[0][1], BoundaryBox, ResolutionX, ResolutionY)
    point_1 = Getint_coords(points[1][0], points[1][1], BoundaryBox, ResolutionX, ResolutionY)
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

def ParseRowColNum(PlotNum, RowNum=None, ColNum=None):
    # @param ColNum: int. Column Number.
    if RowNum in ["Auto", "auto"]:
        RowNum = None
    if ColNum in ["Auto", "auto"]:
        ColNum = None
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
        if PlotNum <= 3:
            return 1, PlotNum
        ColNum = round(PlotNum ** 0.5)
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
    plt.close()
    fig, axes = plt.subplots(nrows=RowNum, ncols=ColNum, figsize=(Width, Height))
    return fig, axes
CreateFigure = CreateCanvasPlt = CreateFigurePlt

def SplitAxRightLeft(ax):
    axLeft = GetSubAx(ax, -0.55, 0.0, 1.0, 1.0)
    axRight = GetSubAx(ax, 0.55, 0.0, 1.0, 1.0)
    return axLeft, axRight

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
    
    if RowNum > 1 or ColNum > 1:
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

def PlotLineChart(ax=None, Xs=None, Ys=None,
        XLabel=None, YLabel=None, Title="Undefined",
        Color="Black", LineWidth=2.0,
        Save=False, SavePath=None,
    ):
    if ax is None:
        fig, ax = plt.subplots()
    Color = ParseColorPlt(Color)
    ax.plot(Xs, Ys, color=Color, linewidth=LineWidth)
    SetXYLabelForAx(ax, XLabel, YLabel)
    SetTitleForAx(ax, Title)
    SaveFigForPlt(Save, SavePath)
    return ax

def PlotBoxPlot(
        ax=None, data=None,
        Norm2Sum1=False, Color="Black", 
        XLabel=None, YLabel=None, Title=None,
        Save=False, SavePath=None,
        **kw
    ):
    data = utils_torch.EnsureFlat(data)
    ax.boxplot(data)
    if Title is not None:
        ax.set_title(Title)
    
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SaveFigForPlt(Save, SavePath)

def PlotHistogram(
        ax=None, data=None,
        Norm2Sum1=False, Color="Black", 
        XLabel=None, YLabel=None, Title=None,
        Save=False, SavePath=None,
        **kw
    ):

    if data is None or data.size == 0:
        ax.text(
            0.5, 0.5, "No Plottable Data", 
            ha='center', va='center',
        )
    else:
        data = utils_torch.EnsureFlat(data)
        data = utils_torch.math.RemoveNaNOrInf(data)
        ax.hist(data, density=Norm2Sum1)
        if Title is not None:
            ax.set_title(Title)
    
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SaveFigForPlt(Save, SavePath)

def SetXYLabelForAx(ax, XLabel, YLabel):
    if XLabel is not None:
        ax.set_xlabel(XLabel)
    if YLabel is not None:
        ax.set_ylabel(YLabel)

def SetTitleForAx(ax, Title):
    if Title is not None:
        ax.set_title(Title)

def GetSubAx(ax, Left, Bottom, Width, Height):
    return ax.inset_axes([Left, Bottom, Width, Height]) # left, bottom, width, height. all are ratios to sub-canvas of ax.

def ParseColorBarOrientation(Orientation):
    if Orientation is None:
        Orientation = "Vertical"
    if Orientation in ["Horizontal", "horizontal"]:
        Orientation = "horizontal"
    elif Orientation in ["Vertical", "vertical"]:
        Orientation = "vertical"
    else:
        raise Exception(Orientation)  
    return Orientation

def PlotColorBarInSubAx(ax, ColorMap="jet", Method="MinMax", Orientation="Vertical", Location=None, **kw):
    # @param Location: [Left, Bottom, Width, Height]
    Orientation = ParseColorBarOrientation(Orientation)
    Location = ParseColorBarLocation(Location, Orientation)
    axSub = GetSubAx(ax, *Location)
    PlotColorBarInAx(axSub, ColorMap=ColorMap, Method=Method, Orientation=Orientation, **kw)

def ParseColorBarLocation(Location, Orientation):
    if Location is None:
        if Orientation in ["vertical"]:
            Location = [1.05, 0.0, 0.05, 1.0]
        elif Orientation in ["horizontal"]:
            Location = [0.0, -0.10, 1.0, 0.05]
        else:
            raise Exception()
    elif isinstance(Location, str):
        raise Exception(Location)
    else:
        pass
    return Location

def PlotColorBarInAx(ax, ColorMap="jet", Method="MinMax", Orientation="Vertical", **kw):
    Orientation = utils_torch.ToLowerStr(Orientation)
    if Orientation not in ["horizontal", "vertical"]:
        raise Exception(Orientation)

    if Method in ["MinMax", "GivenMinMax"]:
        Min, Max = kw["Min"], kw["Max"]
        if Min == Max or not np.isfinite(Min) or not np.isfinite(Max):
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.text(
                0.5, 0.5,"All values are %s"%utils_torch.Float2StrDisplay(Min), 
                rotation=-90.0 if Orientation=="vertical" else 0.0,
                ha='center', va='center',
            )
            ax.axis("off")
            return
    else:
        raise Exception()

    ColorMap = ParseColorMapPlt(ColorMap)
    Norm = mpl.colors.Normalize(vmin=Min, vmax=Max)

    ColorBar = ax.figure.colorbar(
        mpl.cm.ScalarMappable(norm=Norm, cmap=ColorMap),
        cax=ax,
        #ticks=Ticks,
        orientation=Orientation
    )

    Title = kw.get("Title")
    if Title is not None:
        ColorBar.set_label(Title, loc="center")

    Ticks = kw.setdefault("Ticks", "auto")
    SetColorBarTicks(ColorBar, Ticks, Orientation, Min=Min, Max=Max)

    if Orientation in ["horizontal"]: # Avoid Label overlapping
        plt.setp(
            ColorBar.ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor"
        )

def CalculateTickIntervalFloat(Min, Max):
    Range = Max - Min
    if Range == 0.0:
        return 0.0, None, None

    Log = round(math.log(Range, 10))
    Base = 1.0
    Interval = Base * 10 ** Log
    TickNum = Range / Interval

    while not 2.5 <= TickNum <= 6.5:
        if TickNum > 6.0:
            Base, Log = NextIntervalUp(Base, Log)
        elif TickNum < 3.0:
            Base, Log = NextIntervalDown(Base, Log)
        else:
            break
        Interval = Base * 10 ** Log
        TickNum = Range / Interval
    return Interval, Base, Log

def CalculateTickIntervalInt(Min, Max):
    Range = Max - Min
    if Range <= 0:
        return 1
    elif Range <= 5:
        return 2

    Log = round(math.log(Range, 10))
    Base = 1.0
    Interval = Base * 10 ** Log
    TickNum = Range / Interval

    while not 2.5 <= TickNum <= 6.5:
        if TickNum > 6.0:
            Base, Log = NextIntervalUp(Base, Log)
        elif TickNum < 3.0:
            Base, Log = NextIntervalDown(Base, Log)
        else:
            break
        Interval = Base * 10 ** Log
        TickNum = Range / Interval
    return round(Interval)

def NextIntervalUp(Base, Log):
    if Base == 1.0:
        Base = 2.0
    elif Base == 2.0:
        Base = 5.0
    elif Base == 5.0:
        Base = 1.0
        Log += 1
    else:
        raise Exception()
    return Base, Log

def NextIntervalDown(Base, Log):
    if Base == 1.0:
        Base = 5.0
        Log -= 1
    elif Base == 2.0:
        Base = 1.0
    elif Base == 5.0:
        Base = 2.0
    else:
        raise Exception()
    return Base, Log

def CalculateTicksFloat(Method="Auto", Min=None, Max=None, **kw):
    if Method in ["Auto", "auto"]:
        Range = Max - Min
        Interval, Base, Log = CalculateTickIntervalFloat(Min, Max)
        Ticks = []
        Ticks.append(Min)
        Tick = math.ceil(Min / Interval) * Interval
        if Tick - Min < 0.1 * Interval:
            pass
        else:
            Ticks.append(Tick)
        while Tick < Max:
            Tick += Interval
            if Max - Tick < 0.1 * Interval:
                break
            else:
                Ticks.append(Tick)
        Ticks.append(Max)
    elif Method in ["Linear"]:
        Num = kw["Num"]
        Ticks = np.linspace(Min, Max, num=Num)
    else:
        raise Exception()

    TicksStr = []
    if 1 <= Log <= 2:
        TicksStr = list(map(lambda tick:str(int(tick)), Ticks))
    elif Log == 0:
        TicksStr = list(map(lambda tick:'%.1f'%tick, Ticks))
    elif Log == -1:
        TicksStr = list(map(lambda tick:'%.2f'%tick, Ticks))
    elif Log == -2:
        TicksStr = list(map(lambda tick:'%.3f'%tick, Ticks))
    else:
        TicksStr = list(map(lambda tick:'%.2e'%tick, Ticks))
    return Ticks, TicksStr

def CalculateTicksInt(Method="Auto", Min=None, Max=None, **kw):
    assert isinstance(Min, int) and isinstance(Max, int)
    if Method in ["Auto", "auto"]:
        Range = Max - Min
        Interval = CalculateTickIntervalInt(Min, Max)
        Ticks = []
        Ticks.append(Min)
        Tick = round(math.ceil(1.0 * Min / Interval) * Interval)
        if Tick - Min < 0.1 * Interval:
            pass
        else:
            Ticks.append(Tick)
        while Tick < Max:
            Tick += Interval
            if Max - Tick < 0.1 * Interval:
                break
            else:
                Ticks.append(Tick)
        Ticks.append(Max)
    elif Method in ["Linear"]:
        Num = kw["Num"]
        Ticks = np.rint(np.linspace(Min, Max, num=Num))
    else:
        raise Exception()

    TicksStr = list(map(lambda tick:str(int(tick)), Ticks))
    Ticks = list(map(lambda tick:tick - 0.5, Ticks))
    return Ticks, TicksStr


def SetColorBarTicks(ColorBar, Ticks, Orientation, Min=None, Max=None, **kw):
    if not np.isfinite(Min) or not np.isfinite(Max):
        Min, Max = -5.0, 5.0
    if Min==Max:
        if Min == 0.0:
            Min, Max = -1.0, 1.0
        else:
            Min, Max = 0.5 * Min, 1.5 * Max
    if Ticks in ["Auto", "auto"]:
        #Ticks = np.linspace(Min, Max, num=5)
        Ticks, TicksStr = CalculateTicksFloat("Auto", Min, Max)
        ColorBar.set_ticks(Ticks)
        if Orientation in ["vertical"]:
            ColorBar.ax.set_yticklabels(TicksStr)
        elif Orientation in ["horizontal"]:
            ColorBar.ax.set_xticklabels(TicksStr)
        else:
            raise Exception()
    elif Ticks is None: # No Ticks
        pass
    else:
        raise Exception(Ticks)

def SetXTicksFloat(ax, Min, Max, Method="Auto", Rotate45=True):
    if not np.isfinite(Min) or not np.isfinite(Max):
        Min, Max = -5.0, 5.0
    if Min==Max:
        if Min == 0.0:
            Min, Max = -1.0, 1.0
        else:
            Min, Max = 0.5 * Min, 1.5 * Max

    Ticks, TicksStr = CalculateTicksFloat(Method, Min, Max)
    Ticks, TicksStr = CalculateTicksFloat(Method, Min, Max)
    ax.set_xticks(Ticks)
    ax.set_xticklabels(TicksStr)

    if Rotate45: # Avoid Label overlapping
        plt.setp(
            ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor"
        )
    return Ticks, TicksStr

def SetYTicksFloat(ax, Min, Max, Method="Auto"):
    if not np.isfinite(Min) or not np.isfinite(Max):
        Min, Max = -5.0, 5.0
    if Min==Max:
        if Min == 0.0:
            Min, Max = -1.0, 1.0
        else:
            Min, Max = 0.5 * Min, 1.5 * Max
    Ticks, TicksStr = CalculateTicksFloat(Method, Min, Max)
    ax.set_yticks(Ticks)
    ax.set_yticklabels(TicksStr)
    return Ticks, TicksStr

def SetXTicksInt(ax, Min, Max, Method="Auto"):
    if Min > Max:
        Min, Max = Max, Min
    Min, Max = round(Min) + 1, round(Max)
    Ticks, TicksStr = CalculateTicksInt(Method, Min, Max)
    ax.set_xticks(Ticks)
    ax.set_xticklabels(TicksStr)
    return Ticks, TicksStr

def SetYTicksInt(ax, Min, Max, Method="Auto"):
    if Min > Max:
        Min, Max = Max, Min
    Min, Max = round(Min) + 1, round(Max)
    Ticks, TicksStr = CalculateTicksInt(Method, Min, Max)
    ax.set_yticks(Ticks)
    ax.set_yticklabels(TicksStr)
    return Ticks, TicksStr

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

def PlotDistribution1D(data, ):
    data, shape = utils_torch.FlattenData(data)

from scipy.stats import gaussian_kde
def PlotGaussianDensityCurve(
        ax=None, data=None, KernelStd="auto", 
        Save=False, SavePath=None,
        **kw
    ): # For 1D data
    if ax is None:
        fig, ax = plt.subplots()
    data = utils_torch.EnsureFlat(data)
    statistics = utils_torch.math.NpArrayStatistics(data)

    if isinstance(KernelStd, float):
        pass
    elif isinstance(KernelStd, str):
        if KernelStd in ["Auto", "auto"]:
            KernelStd = "Ratio2Std"
        if KernelStd in ["Ratio2Range"]:
            Ratio = kw.setdefault("Ratio", 0.02)
            KernelStd = Ratio * (statistics.Max - statistics.Min)
        elif KernelStd in ["Ratio2Std"]:
            Ratio = kw.setdefault("Ratio", 0.02)
            KernelStd = Ratio * statistics.Std
        else:
            raise Exception()
    else:
        raise Exception(type(KernelStd))
    
    if KernelStd == 0.0:
        KernelStd = 1.0
    try:
        DensityCurve = gaussian_kde(data, bw_method=KernelStd)
        ax.plot(data, DensityCurve(data))
    except Exception:
        # Error when calculating bandwidth for repeating values.
        sns.kdeplot(data)
    # DensityCurve.covariance_factor = lambda : .25
    # DensityCurve._compute_covariance()
    SaveFigForPlt(Save, SavePath)
    return ax

PlotDensityCurve = PlotGaussianDensityCurve

def GetAxRangeMinMax(Min, Max):
    Range = Max - Min
    Left = Min - 0.05 * Range
    Right = Max + 0.05 * Range
    return Left, Right

def SetAxRangeMinMax(ax, Min, Max):
    Range = Max - Min
    Left = Min - 0.05 * Range
    Right = Max + 0.05 * Range
    ax.set_xlim(Left, Right)

def SetMatplotlibParamToDefault():
    mpl.rcParams.update(mpl.rcParamsDefault)


def SetAxisLocationForAx(ax, XAxisLocation=None, YAxisLocation=None):
    if XAxisLocation is not None:
        if XAxisLocation in ["top"]:
            ax.xaxis.tick_top()
        elif XAxisLocation in ["bottom"]:
            pass
        else:
            raise Exception(XAxisLocation)
    if YAxisLocation is not None:
        if YAxisLocation in ["right"]:
            ax.yaxis.tick_right()
        elif YAxisLocation in ["left"]:
            ax.yaxis.tick_left()
        else:
            raise Exception(YAxisLocation)

def SetTitleAndLabelForAx(ax, XLabel=None, YLabel=None, Title=None):
    if XLabel is not None:
        ax.set_xlabel(XLabel)
    if YLabel is not None:
        ax.set_ylabel(YLabel)
    if Title is not None:
        ax.set_title(Title)

def SaveFigForPlt(Save=True, SavePath=None):
    if Save:
        utils_torch.EnsureFileDir(SavePath)
        plt.savefig(SavePath)
        plt.close()

def CompareDensityCurve(data1, data2, Name1, Name2, Save=True, SavePath=None):
    fig, ax = plt.subplots()

    data1 = utils_torch.ToNpArray(data1)
    data2 = utils_torch.ToNpArray(data2)

    Min = min(np.min(data1), np.min(data2))
    Max = max(np.max(data1), np.max(data2))

    utils_torch.plot.PlotGaussianDensityCurve(ax, data1)
    utils_torch.plot.PlotGaussianDensityCurve(ax, data2)

    SetAxRangeMinMax(ax, Min, Max)

    if SavePath is None:
        SavePath = utils_torch.GetSaveDir() + "%s-%s-GaussianKDE.png"%(Name1, Name2)

    utils_torch.plot.SaveFigForPlt(Save, SavePath)

def PlotMeanAndStdAlongTime(
        ax=None, Xs=None, Mean=None, Std=None,
        LineWidth=2.0, Color="Black", XTicks="Int",
        Title=None, XLabel=None, YLabel=None,
        Save=False, SavePath=None, **kw
    ):
    if ax is None:
        fig, ax = plt.subplots()

    if XTicks in ["Int"]:
        XTicks, XTicksStr = SetXTicksInt(ax, min(Xs), max(Xs))
    elif XTicks in ["Float"]:
        XTicks, XTicksStr = SetXTicksFloat(ax, min(Xs), max(Xs))

    Color = ParseColorPlt(Color)
    Y1 = Mean - Std
    Y2 = Mean + Std

    YTicks, YTicksStr = SetYTicksFloat(ax, np.nanmin(Y1), np.nanmax(Y2))

    if utils_torch.math.IsAllNaNOrInf(Y1) and utils_torch.math.IsAllNaNOrInf(Y2) and utils_torch.math.IsAllNaNOrInf(Mean):
        ax.text(
            (XTicks[0] + XTicks[-1]) / 2.0,
            (YTicks[0] + YTicks[-1]) / 2.0,
            "All values are NaN or Inf", ha='center', va='center',
        )
    else:
        PlotLineChart(
            ax, Xs, Mean, Color=Color, LineWidth=LineWidth,
            Save=False, **kw
        )
        ax.fill_between(Xs, Y1, Y2)
    
    

    SetTitleAndLabelForAx(ax, Title, XLabel, YLabel)
    SaveFigForPlt(Save, SavePath)

def SetXAxisLocationForAx(ax, XAXisLocation):
    if XAXisLocation in ["top"]:
        ax.xaxis.tick_top()
    else:
        raise Exception()
