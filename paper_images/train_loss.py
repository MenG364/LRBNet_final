# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import interpolate
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman'] # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
plt.rcParams['axes.unicode_minus'] = False
font = 'times.TTF'  # 字体文件，可使用自己电脑自带的其他字体文件
myfont = fm.FontProperties(fname=font)  # 将所给字体文件转化为此处可以使用的格式

# X轴对应值
x = [i for i in range(0, 22)]  # X轴对应的数值
LRBNet = np.array(
    [None,3.902004004, 3.085659266, 2.854481697, 2.741947412, 2.625797749, 2.533811808, 2.455327988, 2.38549161,
     2.321355343, 2.261657715, 2.067774773, 1.997986555, 1.957150936, 1.924452901, 1.862199426, 1.84551692,
     1.827233195, 1.823233366, 1.81846714, 1.816806316, 1.816460133])
Left = np.array([None,9.8021, 4.4954, 4.1619, 3.9760, 3.8071, 3.6813, 3.5699, 3.4794, 3.4001, 3.3309, 3.2700,
                 3.2132, 3.0462, 2.9887, 2.9537, 2.8929, 2.8769, 2.8587, 2.8545, 2.8493, 2.8464])
right = np.array([None,6.6757, 3.9867, 3.6618, 3.4646, 3.3164, 3.1940, 3.0941, 3.0032, 2.9219, 2.8481, 2.6610,
                  2.6009, 2.5621, 2.5277, 2.4991, 2.4367, 2.4225, 2.4044, 2.3999, 2.3944, 2.3955])
lr = np.array([None,4.0468, 3.1705, 2.9698, 2.8803,2.7757, 2.6858, 2.6072, 2.5364, 2.4745, 2.2813, 2.1674,
                  2.0978, 2.0800, 2.0582, 2.0518, 2.0518, 2.0515, 2.0493, 2.0492, 2.0494, 2.0484])
LRBNet1 = np.array(
    [None,3.7819, 2.9860, 2.854481697, 2.7904, 2.6911, 2.5602, 2.4524, 2.3605,
     2.1913, 2.0795, 2.0148, 1.9773, 1.9546, 1.9372,  1.9273, 1.9205,
     1.9157, 1.9120, 1.9105, 1.9102, 1.9093])
Left1 = np.array([None,10.5763, 4.8977, 4.5081, 4.2999, 4.1215, 3.9776, 3.8612, 3.7613, 3.6715, 3.5911, 3.5175,
                 3.4474, 3.3772, 3.3087, 3.2173, 3.1496, 3.1497, 3.1192, 3.1020, 3.0979, 3.0916])
right1 = np.array([None,7.8078, 4.1464, 3.8274, 3.6820, 3.5014, 3.3518, 3.2217, 3.1092, 3.0027, 2.9071, 2.8160,
                  2.7317, 2.6532, 2.5977, 2.5091, 2.4467, 2.3025, 2.2144, 2.1799, 2.1444, 2.1355])
lr1 = np.array([None,4.0368, 3.2205, 3.0298, 2.9103,2.7957, 2.6858, 2.5972, 2.4364, 2.3345, 2.2613, 2.2274,
                  2.1878, 2.1600, 2.1582, 2.1418, 2.1318, 2.1315, 2.1293, 2.1292, 2.1294, 2.1184])

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
plt.ylim((0, 10))  # y轴的刻度范围被设为a'到b'

# 设置X轴和Y轴的标签名
plt.xlabel('训练轮次', fontsize=18)  # (label, 字体,  字体大小)
plt.ylabel('训练损失',  fontsize=18)

# 将X轴的刻度设为epoch对应的值（也就是只显示epoch对应的那几个刻度值）
del x[0]  # 如果XY轴都是从0开始，可以只保留一个0，另一个删掉

# 设置y轴的显示刻度线的几个值
# ytick = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])    #形式1
ytick = np.arange(start=0, stop=10, step=1)  # [start, stop)   形式2

# 若是想用其他字符(串)代替这些点的值，可以用到它,。比如这个，我将Y轴的0.05、0.15、0.25用空字符串"代替
# ylabel = [0, '', 0.1, '', 0.2, '', 0.3]
# xlabel = []
plt.xticks(x, fontproperties=myfont,fontsize=10)  # 对于X轴，只显示x中各个数对应的刻度值
# plt.yticks(fontproperties=myfont, fontsize=10)  #若是不用ylabel，就用这项
plt.yticks(ytick, fontproperties=myfont,fontsize=10)  # 根据要求决定是否用这个， 显示的刻度值：ylabel的值代替ytick对应的值

# 不显示坐标轴的值
# plt.xticks(())
# plt.yticks(())

ax = plt.gca()  # 返回坐标轴
# 这些都是可选操作
# ax.spines['right'].set_color('none')   #将右边框的颜色设为无
ax.spines['right'].set_visible(False)  #让左边框不可见
ax.spines['top'].set_visible(False)  #让左边框不可见
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
legend_font = {"family": ["SimSun","Times New Roman"]}
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8 , box.height]) #若是将图例画在坐标外边，如果放在右边，一般要给width*0.8左右的值，在上边，要给height*0.8左右的值
ax.legend(loc='upper right', prop=legend_font,fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
          ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)  # 可以用ax.legend,也可以用plt.legend

# 保存图像
plt.savefig('train_loss.svg', dpi=600, bbox_inches='tight',format='svg',edgecolor='black')  #, dpi=100

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
