#!/usr/bin/env python
# coding: utf-8

# In[56]:


#imports
import torch
import cv2
from yolact import Yolact
from collections import defaultdict
import torch.backends.cudnn as cudnn
from utils.augmentations import FastBaseTransform
from utils import timer
from data import cfg
from layers.output_utils import postprocess
from data import COLORS

import tkinter as tk
from tkinter import ttk
import PIL.Image, PIL.ImageTk
from tkinter import BooleanVar, Scale
from PIL import Image, ImageTk
from tkinter.font import Font
from ttkthemes import ThemedStyle


# In[18]:


##SOME PARAMETERS ARGS
global net
global cuda_enabled
global sc
global top_k
global display_text_bbx_enabled
global img_scale
global class_list
global display_label

class_list = set()
cuda_enabled = torch.cuda.is_available()
display_masks_enabled = cuda_enabled
display_label_enabled = True
img_scale=1
sc=0.45
top_k =15
color_cache = defaultdict(lambda: {})
display_text_bbx_enabled=True


# In[3]:


print('Loading model...', end='')
net = Yolact()
net.load_weights("model/yolact_base_54_800000.pth")
net.eval()
print(' Done.')


# In[4]:


def set_default():
    if cuda_enabled and torch.cuda.is_available():
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        cudnn.fastest = False
        torch.set_default_tensor_type('torch.FloatTensor')


# In[5]:


def load_model():
    if cuda_enabled:
        net.cuda()
    else:
        net.cpu()


# In[31]:


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        if cuda_enabled:
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = torch.Tensor(img_numpy).cpu()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0)
        cfg.rescore_bbox = save
    
    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k] 
        
        if cfg.eval_mask_branch:
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]  # Extract classes, scores, and boxes
    num_dets_to_consider = min(top_k, classes.shape[0])
    
    for j in range(num_dets_to_consider):
        if scores[j] < sc:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    if display_masks_enabled and cuda_enabled and cfg.eval_mask_branch and num_dets_to_consider > 0:
        
        masks = masks[:num_dets_to_consider, :, :, None]
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        inv_alph_masks = masks * (-mask_alpha) + 1
        masks_color_summand = masks_color[0]
        
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if num_dets_to_consider == 0:
        return img_numpy

    if display_text_bbx_enabled:
        
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]
            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if display_label_enabled:
                
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if True else _class
                class_list.add(_class)
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy


# In[32]:


def evaluateimg(img):
    
    if cuda_enabled:
        frame = torch.from_numpy(img).cuda().float()
    else:
        frame = torch.from_numpy(img).cpu().float()
        
    batch = FastBaseTransform()(frame.unsqueeze(0))
    
    with torch.no_grad():
        preds = net(batch)

    res = prep_display(preds, frame, None, None, undo_transform=False)
    return res


# In[33]:


def setting():
    set_default()
    load_model()


# In[136]:


class UIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        self.cap = cv2.VideoCapture(0)

        self.label = ttk.Label(root)
        self.label.pack(side="right")

        # Variables
        self.cuda_var = BooleanVar()
        self.display_masks_var = BooleanVar()
        self.display_text_bbx_var = BooleanVar()
        self.display_label_var = BooleanVar()

        self.cuda_var.set(cuda_enabled)
        self.display_masks_var.set(display_masks_enabled)
        self.display_text_bbx_var.set(display_text_bbx_enabled)
        self.display_label_var.set(display_label_enabled)
        self.custom_font = Font(family="Helvetica", size=12, weight="bold", underline=True)
        
        #style = ThemedStyle(self.root)
        #style.set_theme("equilux")  #8
        #style.set_theme("adapta") #9
        
        # Checkboxes
        self.cuda_checkbox = ttk.Checkbutton(self.root, text="CUDA Enable", variable=self.cuda_var, command=self.toggle_cuda)
        self.cuda_checkbox.pack()

        self.display_masks_checkbox = ttk.Checkbutton(self.root, text="Display Mask", variable=self.display_masks_var, command=self.toggle_display_masks)
        self.display_masks_checkbox.pack()

        self.display_text_bbx_checkbox = ttk.Checkbutton(self.root, text="Display B Box", variable=self.display_text_bbx_var, command=self.toggle_display_text_bbx)
        self.display_text_bbx_checkbox.pack()
        
        self.display_label_checkbox = ttk.Checkbutton(self.root, text="Display Label", variable=self.display_label_var, command=self.toggle_display_label)
        self.display_label_checkbox.pack()

        # Sliders
        self.sc_label = ttk.Label(root, text="Select Criteria")
        self.sc_label.pack()
        self.sc_scale = Scale(self.root, from_=0, to=1, resolution=0.01, orient="horizontal", command=self.update_sc)
        self.sc_scale.set(sc)
        self.sc_scale.pack()

        self.img_scale_label = ttk.Label(self.root, text="Image Scale")
        self.img_scale_label.pack()
        
        self.img_scale_scale = Scale(self.root, from_=0.5, to=2, resolution=0.01, orient="horizontal", command=self.update_img_scale)
        self.img_scale_scale.set(img_scale)
        self.img_scale_scale.pack()

        self.top_k_label = ttk.Label(self.root, text="Top K Boxes")
        self.top_k_label.pack()
        
        self.top_k_scale = Scale(self.root, from_=1, to=100, orient="horizontal", command=self.update_top_k)
        self.top_k_scale.set(top_k)
        self.top_k_scale.pack()
        
        self.set_head = ttk.Label(self.root, text="Classes", font=self.custom_font)
        self.set_head.pack()

        self.set_label = ttk.Label(self.root, text="")
        self.set_label.pack()

        self.update_camera_feed()
        self.update_set_label()

    def update_camera_feed(self):
        global class_list
        ret, frame = self.cap.read()

        if ret:
            class_list.clear()
            image = evaluateimg(frame)
            
            #cv2.imshow('output detection', image)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            w, h = img.size
            img = img.resize((int(w*img_scale), int(h*img_scale)))
            img = PIL.ImageTk.PhotoImage(image=img)
            self.label.config(image=img)
            self.label.image = img

        self.root.after(10, self.update_camera_feed)

    def update_set_label(self):
        current_contents = '\n'.join(class_list)  
        self.set_label.config(text=current_contents)
        self.root.after(1000, self.update_set_label)  
        
    def toggle_cuda(self):
        global cuda_enabled
        cuda_enabled = self.cuda_var.get()
        if cuda_enabled:
            self.display_masks_checkbox.config(state="normal") 
        else:
            self.display_masks_checkbox.config(state="disabled")
        setting()

    def update_sc(self,value):
        global sc
        sc = float(value)

    def update_img_scale(self,value):
        global img_scale
        img_scale = float(value)

    def update_top_k(self,value):
        global top_k
        top_k = int(value)

    def toggle_display_masks(self):
        global display_masks_enabled
        display_masks_enabled = self.display_masks_var.get()

    def toggle_display_text_bbx(self):
        global display_text_bbx_enabled
        display_text_bbx_enabled = self.display_text_bbx_var.get()
        
        if display_text_bbx_enabled:
            self.display_label_checkbox.config(state="normal") 
        else:
            self.display_label_checkbox.config(state="disabled")
        setting()

    def toggle_display_label(self):
        global display_label_enabled
        display_label_enabled = self.display_label_var.get()
        setting()


if __name__ == "__main__":
    setting()
    root = tk.Tk()
    #root.configure(bg='white')
    style = ThemedStyle(root)
    #style.set_theme("equilux")  #8
    style.set_theme("adapta") #9
    
    
    app = UIApp(root)
    root.mainloop()
    
    app.cap.release()
    cv2.destroyAllWindows()


# In[133]:


#root.mainloop()


# In[109]:





# In[ ]:




