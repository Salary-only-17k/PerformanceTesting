import cv2
import numpy as np
import random
import os
import base64
import pathlib
import psutil

from Crypto.Cipher import AES
import uuid


def check_recall(ret:int,info:str):
    if ret < 0:
        show_er(info+" failed",f"return-{ret}")
    else:
        show_lg(info+" succed")

class Aescrypt():
    def __init__(self,key,model,iv):
        self.key = self.add_16(key)
        self.model = model
        self.iv = iv

    def add_16(self,par):
        if type(par) == str:
            par = par.encode()
        while len(par) % 16 != 0:
            par += b'\x00'
        return par

    def aesencrypt(self,text):
        text = self.add_16(text)
        if self.model == AES.MODE_CBC:
            self.aes = AES.new(self.key,self.model,self.iv.encode()) 
        elif self.model == AES.MODE_ECB:
            self.aes = AES.new(self.key,self.model) 
        self.encrypt_text = self.aes.encrypt(text)
        return self.encrypt_text

    def aesdecrypt(self,text):
        if self.model == AES.MODE_CBC:
            self.aes = AES.new(self.key,self.model,self.iv.encode()) 
        elif self.model == AES.MODE_ECB:
            self.aes = AES.new(self.key,self.model) 
        self.decrypt_text = self.aes.decrypt(text)
        self.decrypt_text = self.decrypt_text.strip(b"\x00")
        return self.decrypt_text


def get_mac_address():
    mac=uuid.UUID(int = uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e+2] for e in range(0,11,2)])
class ArgParse():
    def __init__(self) -> None:
        base_dir = "configs"
        # base conf
        self.user_conf_pth= f"{base_dir}/user.conf"
        # infer
        self.infer_conf_pth= f"{base_dir}/infer.conf"
        self.test_conf_pth= f"{base_dir}/test.conf"
        self.convert_conf_pth= f"{base_dir}/convert.conf"
   
    def _readConf(self,confPth):
        configDic = dict()
        with open(confPth,'r') as conff:
            for line in conff.readlines():
                if (not line.startswith('#')) and (not line.isspace()):
                    # print(line)
                    key,value = line.split('=')
                    if ',' in value:
                        try:
                            valuelst = [float(v) if '.' in v else int (v) for v in value.strip().split(',')]
                        except:
                            valuelst = [str(v) for v in value.strip().split(',')]
                        configDic[key.strip()]=valuelst
                    else:
                        try:
                            value = float(value.strip()) if '.' in value.strip() else int(value.strip())
                        except:
                            value = value.strip()
                        configDic[key.strip()]=value
        return configDic


    def parse(self):
        checkFileExist(self.user_conf_pth)
        user_params = self._readConf(self.user_conf_pth)
        checkFileExist(self.infer_conf_pth)
        infer_params = self._readConf(self.infer_conf_pth)
        # ，4读reids写stream， 3读redis图片，2读rtsp流，1读视频，0读图片
        if user_params["func"]==0:
            checkFileExist(self.infer_conf_pth)
            infer_params = self._readConf(self.infer_conf_pth)
        else:
            infer_params ={}
        if user_params["func"]==1:
            checkFileExist(self.test_conf_pth)
            test_params = self._readConf(self.test_conf_pth)
        else:
            test_params ={}

        if user_params["func"]==2:
            checkFileExist(self.test_conf_pth)
            test_params = self._readConf(self.test_conf_pth)
        else:
            test_params ={}
        if user_params["func"]==3:
            checkFileExist(self.convert_conf_pth)
            convert_params = self._readConf(self.convert_conf_pth)
        else:
            convert_params ={}
        return {"base_params":user_params,"infer_params":infer_params,"test_params":test_params,"convert_params":convert_params}
    
    

from datetime import datetime

def show_db(n,v=''):
    tm = datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"\033[33m[DEBUG]-[{tm}]  {n}  {v}\033[0m")


def show_lg(n,v=''):
    print(f"\033[34m[INFO]  {n}  {v}\033[0m")


def show_er(n,v='',ex=0):
    print(f"\033[31m[ERROR] {n}  {v}\033[0m")
    if ex<0:
        exit(-1)

def checkFileExist(pth:str):
    if not os.path.exists(pth):
        show_er(pth,"not exist!")
    else:
        show_lg(pth,"reading~")
        
def checkDir(dir:str):
    os.makedirs(dir,exist_ok=True)






class boardInfo():
    def __init__(self):
        show_lg("Development Board Information")
        self.physical_cpu()
        self.physical_memory()
        try:
            self.physical_hard_disk()
        except:
            pass
    def physical_cpu(self):
        """
        获取机器物理CPU个数
        """
        show_lg({"system_cpu_count": psutil.cpu_count(logical=False)})

    def physical_memory(self):
        """
        获取机器物理内存(返回字节bytes)
        """
        show_lg({"system_memory": round(psutil.virtual_memory().total, 2)})

    def physical_hard_disk(self):
        """
        获取机器硬盘信息(字节bytes)
        """
        result = []
        for disk_partition in psutil.disk_partitions():
            o_usage = psutil.disk_usage(disk_partition.device)
            result.append(
                {
                    "device": disk_partition.device,
                    "fstype":disk_partition.fstype,
                    "opts": disk_partition.opts,
                    "total": o_usage.total,
                }
            )
        show_lg({"system_hard_disk": result}) 

def Dct2Lst(dct:dict,ind:int=1):
    if int==0:
        return list(dct.keys())
    else:
        return list(dct.values())
    
def dataPthLst(src:str):
    pth_lst = []
    if os.path.isdir(src):
        for fmt in ["png","jpg","jpeg"]:
            pth_lst += [str(pth) for pth in list(pathlib.Path(src).glob("**/*.{}".format(fmt)))]
    else:
        # os.path.isfile(rk.input_pth) and len(input_format)!=0:
        pth_lst = [src]
    return pth_lst



def saveImg(dir:str,img:np.ndarray):
    cv2.imwrite(os.path.join(dir,f"{random.random():.4f}.jpg"),img)
    

def str2int(v):
    return int(float(v))
def str2float(v):
    return float(v)
def str2bool(v):
    if v=="0" or v ==0:
        return False
    else:
        return True

def image_to_base64(image_np):
        image = cv2.imencode('.jpg',image_np)[1]
        image_code = str(base64.b64encode(image))[2:-1]
        return image_code

def base64_to_image(base64_code):
        img_data = base64.b64decode(base64_code)
        if np.__version__ <"1.18.00":
            img_array = np.fromstring(img_data, np.uint8)
        else:
            img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
        return img
class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self