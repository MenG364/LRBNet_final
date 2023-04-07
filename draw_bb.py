# -*- coding: gbk -*-
import math
import pickle

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.image as mpimg
from matplotlib import ticker
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt


def plot_boxes(img, boxes, att=None, savename=None, scores=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]

        x1 = box[0]
        x2 = box[2]
        y1 = box[1]
        y2 = box[3]

        # if y2 < 0: y2 = box[3] - 1
        rgb = (255, 255, 255)
        font = ImageFont.truetype(
            font='C:\Windows\Fonts\JetBrainsMono-Regular.ttf',
            size=20
        )
        if scores is not None:
            score = round(scores[i], 2)
            red = get_color(2, score, 5)
            green = get_color(1, score, 5)
            blue = get_color(0, score, 5)
            rgb = (red, green, blue)
            draw.text((x1, y1), str(score), fill=rgb, font=font)

        if class_names:
            cls_conf = 1
            cls_id = i
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), str(cls_id), fill=rgb, font=font)
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)
    num_boxes = len(boxes)
    if att is not None:
        from colour import Color
        colors = list(Color("blue").range_to(Color("red"), 100))
        for i in range(num_boxes):
            # i=4
            for j in range(num_boxes):
                if j != i:
                    a = att[i, j].item()
                    index = int(a * 100)
                    rgb = colors[index].get_rgb()
                    rgb = [int(c * 255) for c in rgb]
                    line = ((boxes[i][0] + boxes[i][2]) / 2,
                            (boxes[i][1] + boxes[i][3]) / 2,
                            (boxes[j][0] + boxes[j][2]) / 2,
                            (boxes[j][1] + boxes[j][3]) / 2)
                    draw.line(line, tuple(rgb), 2)

    if savename:
        print("save plot results to %s" % savename)
        # img.save(savename)
        cs = plt.imshow(img, cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
        cb1 = plt.colorbar(cs, fraction=0.03, pad=0.05)
        cb1.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cb1.update_ticks()

        plt.savefig(savename, bbox_inches='tight', dpi=600)
        plt.show()

    return img


def draw_attmap(att, savename, ax=None, vmin=0, vmax=1,fontsize=20):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    y = att.size(0)
    x = att.size(1)
    cs = plt.imshow(att.permute(1, 0).tolist(), cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax, origin='lower')
    cb = plt.colorbar(cs, fraction=0.03, pad=0.05)
    cb.ax.tick_params(labelsize=fontsize)
    plt.xticks(np.arange(0, y, 1), np.arange(0, y, 1) if ax == None else ax,size=fontsize)
    plt.yticks(np.arange(0, x, 1), "Is this a motorcycle or bike".split(" "),size=20)
    plt.xlabel("区域对象", size=fontsize)
    plt.ylabel("问题单词", size=fontsize)
    plt.gca().invert_yaxis()
    plt.title("视觉嵌入分类器",size=fontsize)
    plt.savefig(savename, bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    imgid = [42, 73]
    question_id = [42000, 42001, 42002, 73000, 73001, 73002, 73003, 74000]
    questions = pickle.load(open('result/questions.pkl', 'rb'))
    idx = 2
    question = questions[idx]
    img = 'F:\datasets\\val2014\\COCO_val2014_000000000042.jpg'
    lena = Image.open(img).convert('RGB')
    boxes = pickle.load(open('result/bb.pkl', 'rb'))
    num_bb = 3
    box = list(boxes[idx])[:num_bb]
    class_names = pickle.load(open('result/captions.pkl', 'rb'))
    class_name = class_names[idx]
    ##########################################################################

    # atts_L = pickle.load(open('result/imp_adj_mat_L.pkl', 'rb'))
    # atts_R = pickle.load(open('result/imp_adj_mat_R.pkl', 'rb'))
    # att_L = np.array(atts_L[0].cpu().squeeze())[:num_bb, :num_bb]
    # att_R = np.array(atts_R[0].cpu().squeeze())[:num_bb, :num_bb]
    # # plot_boxes(lena, box[1:6], savename='result/LRBNet_attmap/imp_adj_mat_R_42.jpg', att=att[1:,1:],class_names=class_name)
    # plot_boxes(lena, box, savename='result/LRBNet_attmap/imp_adj_mat_L_73.jpg', att=att_L, class_names=class_name)
    # plot_boxes(lena, box, savename='result/LRBNet_attmap/imp_adj_mat_R_73.jpg', att=att_R, class_names=class_name)
    ################################################################################
    atts = pickle.load(open('result/s-dmls/att/att_tensor(73001).pkl', 'rb'))[2]
    # atts = atts.flip(3)
    att = atts[idx, :10, :6]

    eps = 1e-8
    # att = att.permute(0, 2, 1)[:, :7]
    att = (att + eps).log()
    vmin = torch.min(att).item()
    vmax = torch.max(att).item()
    # for i in range(8):
    # draw_attmap(att[i], 'result/LRBNet_attmap/v_att_map_' + str(i) + '_.jpg', ax=question, vmin=vmin, vmax=vmax)
    # draw_attmap(att[i], 'result/LRBNet_attmap/joint_att_map_' + str(i) + '_.png',vmin=vmin,vmax=vmax)
    draw_attmap(att, 'result/s-dmls/img/att_tensor(73001)r.png', vmin=-18, vmax=0)
    # for i in range(16):
    #     draw_attmap(att[i], 'result/LRBNet_attmap/cap_att_map_' + str(i) + '_.png', ax=question,vmin=vmin,vmax=vmax)
    # draw_attmap(att[2,:num_bb,:], 'result/s-dmls/img/att' + str(2) + '_.png',vmin=vmin,vmax=vmax)
    # att=att.sum(1)/16
    # draw_attmap(att[:num_bb,:num_bb], 'result/LRBNet_attmap/aff_softmax_l.png', vmin=0, vmax=1)
