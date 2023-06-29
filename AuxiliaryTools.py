import pathlib
import os
from typing import Any
import cv2

class gender_datatxt():
    def __init__(self,pth,flg,datap) -> None:
        self.flg =flg
        self.pth = pth
        
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.flg ==0:
            self.img2txt()
        else:
            self.video2txt()
    def img2txt(self):
        datap = os.path.join("data",datap)
        pth_lst = []
        for fmt in ['.jpg','png','.jpeg','.bmp']:
            pth_lst += list(pathlib.Path(self.pth).glob(f"**/*{fmt}"))
        with open(datap,'w') as f:
            for p in pth_lst:
                f.write(str(p)+'\n')
    def video2txt(self):
        
        n=1
        cap = cv2.VideoCapture(self.pth)
        base = os.path.join("runs",os.path.basename(self.pth))
        datap = os.path.join("data",os.path.basename(self.pth),datap)
        os.makedirs(base,exist_ok=True)
        while cap.Isopen():
            ret,frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(base,f"{n:0>3d}.jpg"))
        pth_lst = []
        pth_lst += list(pathlib.Path(base).glob("**/*.jpg"))
        with open(datap,'w') as f:
            for p in pth_lst:
                f.write(str(p)+'\n')
        