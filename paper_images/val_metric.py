# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import interpolate

plt.rcParams['font.family'] = ['SimSun', 'Times New Roman'] # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
# plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False
font = 'times.TTF'  # 字体文件，可使用自己电脑自带的其他字体文件
myfont = fm.FontProperties(fname=font)  # 将所给字体文件转化为此处可以使用的格式

# X轴对应值
x = [i for i in range(0, 22)]  # X轴对应的数值
LRBNet = np.array(
    [None, 57.23086929, 62.63867188, 64.13638306, 64.34448242, 65.24114227, 65.52194977, 65.69377899, 65.77626038,
     65.79451752, 65.72904205, 66.32701111, 66.34707642, 66.38967133, 66.29766083, 66.33589172, 66.31156921,
     66.34635925, 66.33854675, 66.31472778, 66.3210144, 66.32491302])
Left = np.array(
    [None, 42.9848, 46.4638, 48.5192, 50.0267, 51.8633, 53.9086, 55.0496, 56.2545, 57.2775, 57.6536, 58.5773,
     58.4847, 59.4818, 59.6559, 59.7213, 59.8713, 59.879, 59.9459, 59.9364, 59.8970, 59.9558])
right = np.array(
    [None, 49.1578, 54.3370, 57.3965, 59.2405, 60.6888, 61.6274, 62.6313, 62.9035, 63.7425, 63.6221, 64.7534,
     64.9078, 64.9933, 65.0797, 65.0711, 65.3702, 65.3141, 65.3466, 65.4031, 65.3686, 65.3637])
lr = np.array([None, 57.72, 62.9375,  63.8714,64.5030, 64.8295, 65.0135, 65.4767,  65.6353, 65.6292, 66.1847,
               66.2067,66.1835, 66.2169, 66.2131, 66.2149,66.2219, 66.2138, 66.2157, 66.1951, 66.2065, 66.1999])
LRBNet1 = np.array(
    [None, 60.3022, 63.6189, 64.2013, 64.4233, 64.6481, 64.8216, 64.6133,65.7281,
     65.7010, 66.0074, 66.0305, 66.0006, 66.0669, 66.0925, 66.0908, 66.0006, 66.0669, 66.0925, 66.0908, 66.0669, 66.0925])
Left1 = np.array(
    [None, 36.2217, 39.9032, 42.0176, 43.9987, 45.0546, 46.9855, 47.8608, 48.68655,49.2941, 49.9265,50.4029,
     50.5551, 50.4258, 51.2931, 51.2462, 51.7414, 51.8205, 51.8148, 51.8833, 51.8884, 51.9042])
right1 = np.array(
    [None, 48.3516, 53.0939, 55.3481, 56.6250, 58.7149, 60.0729, 61.1595, 61.7141 , 62.4980, 62.9920, 63.1247,
     63.4020, 63.4305, 63.5957,  63.7384, 63.6401, 64.7592, 64.7336,  65.1951,  65.1951, 65.2637])
lr1 = np.array([None, 59.32, 62.7975,  63.6714,63.8330, 64.0895, 64.2335, 64.1867,  65.2953, 65.3292, 65.2347,
               65.6567,65.6935, 65.6869, 65.7531, 65.7649,65.7619, 65.7619, 65.7619, 65.7419, 65.7319, 65.7319])
# Y轴对应值，要保证X和Y轴数值的个数相等
# baidu = np.array([8.5, 8.8, 10.7, 8.2, 8.5, 10.6, 8, 7.4]) / 100
# _360 = np.array([5.5, 22.4, 19.8, 20.9, 18.2, 16.9, 16.5, 20.9]) / 100
# alipay = np.array([13.2, 24.3, 22.6, 24.1, 27.9, 27.6, 27.1, 30.6]) / 100
# yy = np.array([8.2, 14.3, 15.6, 22.1, 25.9, 27.6, 28.1, 31.6])/100

# 若是想平滑某一折线，可以参考这部分
# xnew =np.arange(0, 10000, 100)
# func = interpolate.interp1d(x, yy, kind='cubic')  #quadratic、cubic、slinear
# ynew = func(xnew)
# markevery = []
# for i, j in enumerate(xnew):
#     if(j in x):
#         markevery.append(i)

plt.figure(facecolor='#FFFFFF', figsize=(8, 10))  # 将图的外围设为白色

# 图的标题   参数（图的标题，字体，字号， 颜色， 位置），注：颜色取值在0到1之间
# plt.title('The influence of the degree of different basic model training on the effect of migration learning',
#           fontproperties=myfont, fontsize=22, color=(0.4,0.4,0.4), loc='center')

# 画线        参数（X轴数组， Y轴数组， 折点的标记， 线条的类型， 每条线所表示的标签， 线的粗细， 折点标记大小，  折点标记颜色， 标记的边缘线条颜色， 标记边缘线条粗细）
# 标记marker可取值：'.‘、 ','、'o'、'v'、'^'、'<'、'>'、'1'、'2'、'3'、'4'、's'、'p'、'*'、'h'、'H'、'+'、'x'、'D'、'd'、'|'、'_'等
color_dict = {
    'S-DMLS': '#3120E0',
    'S-TUM': '#0078AA',
    'S-VUM': '#3AB4F2',
    'S-TUM+VUM': '#FFB319',
    'G-DMLS': '#C21010',
    'G-TUM': '#E64848',
    'G-VUM': '#21E1E1',
    'G-TUM+VUM': '#125C13'
}  # 用于设置线条的备选颜色
linewidth=2.0
cl = color_dict
plt.plot(x, LRBNet,  ls='-', label=r'$\rm G-DMLS$'+'（本文方法）', c=cl['G-DMLS'], linewidth=linewidth, ms=6, mfc=cl['G-DMLS'],
         mec=cl['G-DMLS'], mew=3, mfcalt='m')
plt.plot(x, Left,  ls='-', label=r'$\rm G-TUM$', c=cl['G-TUM'], linewidth=linewidth, ms=6, mfc=cl['G-TUM'], mec=cl['G-TUM'],
         mew=3)
plt.plot(x, right,  ls='-', label=r'$\rm G-VUM$', c=cl['G-VUM'], linewidth=linewidth, ms=6, mfc=cl['G-VUM'],
         mec=cl['G-VUM'], mew=3)
plt.plot(x, lr,  ls='-', label=r'$\rm G-TUM+VUM$', c=cl['G-TUM+VUM'], linewidth=linewidth, ms=6, mfc=cl['G-TUM+VUM'],
         mec=cl['G-TUM+VUM'], mew=3)

plt.plot(x, LRBNet1,  ls='-', label=r'$\rm S-DMLS$'+'（本文方法）', c=cl['S-DMLS'], linewidth=linewidth, ms=6, mfc=cl['S-DMLS'],
         mec=cl['S-DMLS'], mew=3, mfcalt='m')
plt.plot(x, Left1,  ls='-', label=r'$\rm S-TUM$', c=cl['S-TUM'], linewidth=linewidth, ms=6, mfc=cl['S-TUM'], mec=cl['S-TUM'],
         mew=3)
plt.plot(x, right1,  ls='-', label=r'$\rm S-VUM$', c=cl['S-VUM'], linewidth=linewidth, ms=6, mfc=cl['S-VUM'],
         mec=cl['S-VUM'], mew=3)
plt.plot(x, lr1,  ls='-', label=r'$\rm S-TUM+VUM$', c=cl['S-TUM+VUM'], linewidth=linewidth, ms=6, mfc=cl['S-TUM+VUM'],
         mec=cl['S-TUM+VUM'], mew=3)
# 平滑折线的示例
# plt.plot(xnew, ynew, marker='.', markevery=markevery, ls='-', label='yy', c='purple', linewidth=1.0, ms=6, mfc='purple', mec='purple', mew=3)

# 设置横纵坐标的刻度范围
plt.xlim((0, 20))  # x轴的刻度范围被设为a到b
plt.ylim((45, 67))  # y轴的刻度范围被设为a'到b'

# 设置X轴和Y轴的标签名
plt.xlabel('训练轮次', fontsize=18)  # (label, 字体,  字体大小)
plt.ylabel(r'验证集准确度（$\rm \%$）', fontsize=18)

# 将X轴的刻度设为epoch对应的值（也就是只显示epoch对应的那几个刻度值）
del x[0]  # 如果XY轴都是从0开始，可以只保留一个0，另一个删掉

# 设置y轴的显示刻度线的几个值
# ytick = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])    #形式1
ytick = np.arange(start=45, stop=67, step=2)  # [start, stop)   形式2

# 若是想用其他字符(串)代替这些点的值，可以用到它,。比如这个，我将Y轴的0.05、0.15、0.25用空字符串"代替
# ylabel = [i for i in range(40,2,70)]
# xlabel = []
plt.xticks(x, fontproperties=myfont,fontsize=10)  # 对于X轴，只显示x中各个数对应的刻度值
# plt.yticks(fontproperties=myfont,fontsize=10)  #若是不用ylabel，就用这项
plt.yticks(ytick, fontproperties=myfont,fontsize=10)  # 根据要求决定是否用这个， 显示的刻度值：ylabel的值代替ytick对应的值

# 不显示坐标轴的值
# plt.xticks(())
# plt.yticks(())

ax = plt.gca()  # 返回坐标轴
# 这些都是可选操作
# ax.spines['right'].set_color('none')   #将右边框的颜色设为无
ax.spines['right'].set_visible(False)  # 让左边框不可见
ax.spines['top'].set_visible(False)  # 让左边框不可见
# ax.spines['top'].set_color('none')   #将上边框设为无色
# ax.spines['left'].set_color('green')   #将左边框的颜色设为绿色
# ax.xaxis.set_ticks_position('bottom') #将x轴的刻度值写在下边框的下面
# ax.yaxis.set_ticks_position('left')  #将y轴的刻度值写左边框的左面
# ax.spines['bottom'].set_position(('data', 0))   #将下边框的位置挪到y轴的0刻度处
# ax.spines['left'].set_position(('data', 50))  #将左边框挪到x轴的50刻度处

ax.tick_params(axis='x', tickdir='in')  # 坐标轴的刻度线及刻度值的一些参数设置，详见文末
ax.tick_params(axis='y', tickdir='in')

# 可选部分
axlinewidth = 1.2
axcolor = '#4F4F4F'
ax.spines['bottom'].set_linewidth(axlinewidth)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(axlinewidth)  # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(axlinewidth)  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(axlinewidth)  # 设置上部坐标轴的粗细
ax.spines['bottom'].set_color(axcolor)  # 设置坐标轴颜色
ax.spines['left'].set_color(axcolor)
ax.spines['right'].set_color(axcolor)
ax.spines['top'].set_color(axcolor)

# 显示图例
'''loc:图例位置， 
   fontsize：字体大小，
   frameon：是否显示图例边框，
   ncol：图例的列的数量，一般为1,
   title:为图例添加标题
   shadow:为图例边框添加阴影,
   markerfirst:True表示图例标签在句柄右侧，false反之，
   markerscale：图例标记为原图标记中的多少倍大小，
   numpoints:表示图例中的句柄上的标记点的个数，一半设为1,
   fancybox:是否将图例框的边角设为圆形
   framealpha:控制图例框的透明度
   borderpad: 图例框内边距
   labelspacing: 图例中条目之间的距离
   handlelength:图例句柄的长度
   bbox_to_anchor: (横向看右，纵向看下),如果要自定义图例位置或者将图例画在坐标外边，用它，比如bbox_to_anchor=(1.4,0.8)，这个一般配合着ax.get_position()，set_position([box.x0, box.y0, box.width*0.8 , box.height])使用
   用不到的参数可以直接去掉,有的参数没写进去，用得到的话加进去     , bbox_to_anchor=(1.11,0)
'''
# legend_font = {"family" : ["SimSun","Times New Roman"]}
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8 , box.height]) #若是将图例画在坐标外边，如果放在右边，一般要给width*0.8左右的值，在上边，要给height*0.8左右的值
ax.legend(loc='lower right',fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
          ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)  # 可以用ax.legend,也可以用plt.legend

# 保存图像
plt.savefig('val_metric.svg', dpi=600, bbox_inches='tight', format='svg', edgecolor='black')  # , dpi=100

# 显示图
plt.show()

'''
plt.legend(loc)中的loc取值：
 0:'best'
 1: 'upper right'
 2: 'upper left'
 3:'lower left'
 4: 'lower right'
 5: 'right'
 6: 'center left'
 7: 'center right'
 8: 'lower center'
 9: 'upper center'
 10: 'center'
'''

'''
Axes.tick_params(axis='both', **kwargs)
参数:
axis :  {'x', 'y', 'both'} 选择对哪个轴操作，默认是'both'
reset :  bool If True, set all parameters to defaults before processing other keyword arguments. Default is False.
which :  {'major', 'minor', 'both'} 选择对主or副坐标轴进行操作
direction/tickdir : {'in', 'out', 'inout'}刻度线的方向
size/length : float, 刻度线的长度
width :  float, 刻度线的宽度
color :  刻度线的颜色
pad :  float, 刻度线与刻度值之间的距离
labelsize :  float/str, 刻度值字体大小
labelcolor : 刻度值颜色
colors :  同时设置刻度线和刻度值的颜色
zorder : float Tick and label zorder.
bottom, top, left, right : bool, 分别表示上下左右四边，是否显示刻度线，True为显示
labelbottom, labeltop, labelleft, labelright :bool, 分别表示上下左右四边，是否显示刻度值，True为显示
labelrotation : 刻度值逆时针旋转给定的度数，如20
gridOn: bool ,是否添加网格线； grid_alpha:float网格线透明度 ； grid_color: 网格线颜色;  grid_linewidth:float网格线宽度； grid_linestyle: 网格线型 
tick1On, tick2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度线
label1On,label2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度值

ALL param:
['size', 'width', 'color', 'tickdir', 'pad', 'labelsize', 'labelcolor', 'zorder', 'gridOn', 'tick1On', 'tick2On', 
'label1On', 'label2On', 'length', 'direction', 'left', 'bottom', 'right', 'top', 'labelleft', 'labelbottom', 'labelright',
 'labeltop', 'labelrotation', 'grid_agg_filter', 'grid_alpha', 'grid_animated', 'grid_antialiased', 'grid_clip_box', 
 'grid_clip_on', 'grid_clip_path', 'grid_color', 'grid_contains', 'grid_dash_capstyle', 'grid_dash_joinstyle', 'grid_dashes', 
 'grid_drawstyle', 'grid_figure', 'grid_fillstyle', 'grid_gid', 'grid_label', 'grid_linestyle', 'grid_linewidth', 'grid_marker', 
 'grid_markeredgecolor', 'grid_markeredgewidth', 'grid_markerfacecolor', 'grid_markerfacecoloralt', 'grid_markersize', 
 'grid_markevery', 'grid_path_effects', 'grid_picker', 'grid_pickradius', 'grid_rasterized', 'grid_sketch_params', 'grid_snap', 
 'grid_solid_capstyle', 'grid_solid_joinstyle', 'grid_transform', 'grid_url', 'grid_visible', 'grid_xdata', 'grid_ydata', 
 'grid_zorder', 'grid_aa', 'grid_c', 'grid_ls', 'grid_lw', 'grid_mec', 'grid_mew', 'grid_mfc', 'grid_mfcalt', 'grid_ms']
'''
