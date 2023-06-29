import argparse
import os
from common.api import cls_stream,det_stream,otr_stream,seg_stream
from common.utils import show_lg
# from PerformanceTesting.common.stream import Pipe
# from common.utils import show_lg

def parse_opt():
    __doc__ = """  # [function] 0-导出模型为rknn  1-推理  2-性能评估,例如 0, 或 0,1
        self.func = 1,
        量化校准图像列表
        self.txt_pth = "data/coco.txt"
        可视化路径"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=int, nargs='+',required=True, help='run mode,0,1,2,3,4') 
    parser.add_argument('--mp', type=str, required=True, help='model path') 
    opt = parser.parse_args()
    assert os.path.exists(opt.mp),f"{opt.mp} not exist"
    if opt.mp.endswith(".rknn"):
        try: 
            opt.mode.remove(0)
        except:
            pass
    
    return opt



if __name__ == "__main__":
    show_lg("~"*26)
    opt = parse_opt()
    show_lg("~"*7+">>>params<<<"+"~"*7)
    show_lg(opt)
    show_lg("~"*26)
    # 根据自己的事迹情况，选择不同的功能 分类或者检测
    try:
        f_api = cls_stream(opt.mp)
    except:
        f_apt = det_stream(opt.mp)
    show_lg("~"*26)
    f_api.env_info()
    if 0 in opt.mode:
        f_api.export()
    if 1 in opt.mode:
        f_api.infer()
    if 2 in opt.mode:
        f_api.test()
    show_lg("done")
    